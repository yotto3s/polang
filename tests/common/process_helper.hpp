#ifndef POLANG_TEST_PROCESS_HELPER_HPP
#define POLANG_TEST_PROCESS_HELPER_HPP

#include <array>
#include <cstdlib>
#include <string>
#include <sys/wait.h>
#include <unistd.h>
#include <vector>

struct ProcessResult {
  std::string stdout_output;
  std::string stderr_output;
  int exit_code;
};

inline ProcessResult runProcess(const std::string& executable,
                                const std::string& input) {
  ProcessResult result;
  result.exit_code = -1;

  // Create pipes for stdin, stdout, stderr
  int stdin_pipe[2];
  int stdout_pipe[2];
  int stderr_pipe[2];

  if (pipe(stdin_pipe) == -1 || pipe(stdout_pipe) == -1 ||
      pipe(stderr_pipe) == -1) {
    return result;
  }

  const pid_t pid = fork();

  if (pid == -1) {
    // Fork failed
    return result;
  }

  if (pid == 0) {
    // Child process
    close(stdin_pipe[1]);  // Close write end of stdin
    close(stdout_pipe[0]); // Close read end of stdout
    close(stderr_pipe[0]); // Close read end of stderr

    dup2(stdin_pipe[0], STDIN_FILENO);
    dup2(stdout_pipe[1], STDOUT_FILENO);
    dup2(stderr_pipe[1], STDERR_FILENO);

    close(stdin_pipe[0]);
    close(stdout_pipe[1]);
    close(stderr_pipe[1]);

    execlp(executable.c_str(), executable.c_str(), nullptr);
    _exit(127); // exec failed
  }

  // Parent process
  close(stdin_pipe[0]);  // Close read end of stdin
  close(stdout_pipe[1]); // Close write end of stdout
  close(stderr_pipe[1]); // Close write end of stderr

  // Write input to child's stdin
  write(stdin_pipe[1], input.c_str(), input.size());
  close(stdin_pipe[1]);

  // Read stdout
  std::array<char, 4096> buffer{};
  ssize_t bytes_read = 0;
  while ((bytes_read = read(stdout_pipe[0], buffer.data(), buffer.size())) >
         0) {
    result.stdout_output.append(buffer.data(),
                                static_cast<std::size_t>(bytes_read));
  }
  close(stdout_pipe[0]);

  // Read stderr
  while ((bytes_read = read(stderr_pipe[0], buffer.data(), buffer.size())) >
         0) {
    result.stderr_output.append(buffer.data(),
                                static_cast<std::size_t>(bytes_read));
  }
  close(stderr_pipe[0]);

  // Wait for child to finish
  int status = 0;
  waitpid(pid, &status, 0);

  if (WIFEXITED(status)) {
    result.exit_code = WEXITSTATUS(status);
  }

  return result;
}

inline ProcessResult runCompiler(const std::string& source) {
  return runProcess(POLANG_COMPILER_PATH, source);
}

inline ProcessResult runProcessWithArgs(const std::string& executable,
                                        const std::vector<std::string>& args) {
  ProcessResult result;
  result.exit_code = -1;

  int stdout_pipe[2];
  int stderr_pipe[2];

  if (pipe(stdout_pipe) == -1 || pipe(stderr_pipe) == -1) {
    return result;
  }

  const pid_t pid = fork();

  if (pid == -1) {
    return result;
  }

  if (pid == 0) {
    // Child process
    close(stdout_pipe[0]);
    close(stderr_pipe[0]);

    dup2(stdout_pipe[1], STDOUT_FILENO);
    dup2(stderr_pipe[1], STDERR_FILENO);

    close(stdout_pipe[1]);
    close(stderr_pipe[1]);

    // Build argv array
    std::vector<char*> argv;
    argv.push_back(const_cast<char*>(executable.c_str()));
    for (const auto& arg : args) {
      argv.push_back(const_cast<char*>(arg.c_str()));
    }
    argv.push_back(nullptr);

    execv(executable.c_str(), argv.data());
    _exit(127);
  }

  // Parent process
  close(stdout_pipe[1]);
  close(stderr_pipe[1]);

  std::array<char, 4096> buffer{};
  ssize_t bytes_read = 0;
  while ((bytes_read = read(stdout_pipe[0], buffer.data(), buffer.size())) >
         0) {
    result.stdout_output.append(buffer.data(),
                                static_cast<std::size_t>(bytes_read));
  }
  close(stdout_pipe[0]);

  while ((bytes_read = read(stderr_pipe[0], buffer.data(), buffer.size())) >
         0) {
    result.stderr_output.append(buffer.data(),
                                static_cast<std::size_t>(bytes_read));
  }
  close(stderr_pipe[0]);

  int status = 0;
  waitpid(pid, &status, 0);

  if (WIFEXITED(status)) {
    result.exit_code = WEXITSTATUS(status);
  }

  return result;
}

inline ProcessResult runCompilerWithArgs(const std::vector<std::string>& args) {
  return runProcessWithArgs(POLANG_COMPILER_PATH, args);
}

inline ProcessResult runRepl(const std::string& source) {
  // REPL finds PolangCompiler in same directory automatically
  return runProcess(POLANG_REPL_PATH, source);
}

#endif // POLANG_TEST_PROCESS_HELPER_HPP
