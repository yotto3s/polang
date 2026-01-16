#include <llvm/AsmParser/Parser.h>
#include <llvm/ExecutionEngine/Orc/LLJIT.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/Support/SourceMgr.h>
#include <llvm/Support/TargetSelect.h>
#include <llvm/Support/raw_ostream.h>

#include <cstring>
#include <iostream>
#include <sstream>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>

using namespace llvm;

// Path to PolangCompiler (empty means use PATH search)
static std::string compilerPath;

// Check if file exists and is executable
static bool isExecutable(const std::string& path) {
  struct stat st;
  if (stat(path.c_str(), &st) != 0) {
    return false;
  }
  return (st.st_mode & S_IXUSR) != 0;
}

// Get directory containing the executable
static std::string getExeDirectory(const char* argv0) {
  // Try /proc/self/exe first (Linux)
  char buf[4096];
  const ssize_t len = readlink("/proc/self/exe", buf, sizeof(buf) - 1);
  std::string exePath;
  if (len > 0) {
    buf[len] = '\0';
    exePath = buf;
  } else {
    // Fall back to argv[0]
    exePath = argv0;
  }

  const size_t pos = exePath.rfind('/');
  if (pos == std::string::npos) {
    return ".";
  }
  return exePath.substr(0, pos);
}

// Initialize compiler path - search in exe directory first, then fall back to
// PATH
static void initCompilerPath(const char* argv0) {
  const std::string exeDir = getExeDirectory(argv0);
  const std::string localCompiler = exeDir + "/PolangCompiler";

  if (isExecutable(localCompiler)) {
    compilerPath = localCompiler;
  }
  // If not found, compilerPath remains empty and we'll use execlp (PATH search)
}

// Run PolangCompiler as subprocess
// If filename is provided, pass it as argument; otherwise pipe source via stdin
static std::string runCompiler(const std::string& source,
                               const char* filename = nullptr) {
  int pipeIn[2];  // Parent writes, child reads (child's stdin)
  int pipeOut[2]; // Child writes, parent reads (child's stdout)

  if (pipe(pipeIn) < 0 || pipe(pipeOut) < 0) {
    std::cerr << "Failed to create pipes\n";
    return "";
  }

  const pid_t pid = fork();
  if (pid < 0) {
    std::cerr << "Failed to fork\n";
    return "";
  }

  if (pid == 0) {
    // Child process
    close(pipeIn[1]);  // Close write end of input pipe
    close(pipeOut[0]); // Close read end of output pipe

    dup2(pipeIn[0], STDIN_FILENO);
    dup2(pipeOut[1], STDOUT_FILENO);

    close(pipeIn[0]);
    close(pipeOut[1]);

    // Execute PolangCompiler
    if (!compilerPath.empty()) {
      // Use compiler found in same directory as REPL
      if (filename != nullptr) {
        execl(compilerPath.c_str(), "PolangCompiler", filename, nullptr);
      } else {
        execl(compilerPath.c_str(), "PolangCompiler", nullptr);
      }
    } else {
      // Fall back to PATH search
      if (filename != nullptr) {
        execlp("PolangCompiler", "PolangCompiler", filename, nullptr);
      } else {
        execlp("PolangCompiler", "PolangCompiler", nullptr);
      }
    }

    // If exec fails, exit
    std::cerr << "Failed to exec PolangCompiler\n";
    _exit(1);
  }

  // Parent process
  close(pipeIn[0]);  // Close read end of input pipe
  close(pipeOut[1]); // Close write end of output pipe

  // Write source to compiler's stdin only in stdin mode
  if (filename == nullptr) {
    write(pipeIn[1], source.c_str(), source.size());
  }
  close(pipeIn[1]); // Signal EOF

  // Read IR from compiler's stdout
  std::string ir;
  char buffer[4096];
  ssize_t n;
  while ((n = read(pipeOut[0], buffer, sizeof(buffer))) > 0) {
    ir.append(buffer, n);
  }
  close(pipeOut[0]);

  // Wait for child to finish
  int status;
  waitpid(pid, &status, 0);

  if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) {
    std::cerr << "Compiler failed\n";
    return "";
  }

  return ir;
}

// Parse IR string and execute with JIT
static int executeIR(const std::string& ir) {
  InitializeNativeTarget();
  InitializeNativeTargetAsmPrinter();

  LLVMContext context;
  SMDiagnostic err;

  // Parse the IR string
  std::unique_ptr<Module> module = parseAssemblyString(ir, err, context);
  if (!module) {
    err.print("PolangRepl", errs());
    return 1;
  }

  // Create JIT
  auto JTMB = orc::JITTargetMachineBuilder::detectHost();
  if (!JTMB) {
    std::cerr << "Failed to detect host: " << toString(JTMB.takeError())
              << "\n";
    return 1;
  }
  JTMB->setCPU("generic");

  auto JIT =
      orc::LLJITBuilder().setJITTargetMachineBuilder(std::move(*JTMB)).create();
  if (!JIT) {
    std::cerr << "Failed to create JIT: " << toString(JIT.takeError()) << "\n";
    return 1;
  }

  // Add module to JIT - create a new context for JIT execution
  auto jitContext = std::make_unique<LLVMContext>();

  // Re-parse in the new context for JIT
  std::unique_ptr<Module> jitModule = parseAssemblyString(ir, err, *jitContext);
  if (!jitModule) {
    err.print("PolangRepl", errs());
    return 1;
  }

  auto TSM = orc::ThreadSafeModule(std::move(jitModule), std::move(jitContext));

  if (auto Err = (*JIT)->addIRModule(std::move(TSM))) {
    std::cerr << "Failed to add module: " << toString(std::move(Err)) << "\n";
    return 1;
  }

  // Lookup and run main
  auto MainSym = (*JIT)->lookup("main");
  if (!MainSym) {
    std::cerr << "Failed to find main: " << toString(MainSym.takeError())
              << "\n";
    return 1;
  }

  auto* MainFn = MainSym->toPtr<void (*)()>();
  MainFn();

  return 0;
}

int main(int argc, char** argv) {
  // Find PolangCompiler in same directory as REPL, or fall back to PATH
  initCompilerPath(argv[0]);

  std::string ir;

  if (argc > 1) {
    // File input mode - pass filename to compiler
    ir = runCompiler("", argv[1]);
  } else {
    // Stdin mode
    std::stringstream buffer;
    buffer << std::cin.rdbuf();
    const std::string source = buffer.str();
    ir = runCompiler(source, nullptr);
  }

  if (ir.empty()) {
    return 1;
  }

  // Execute the IR
  return executeIR(ir);
}
