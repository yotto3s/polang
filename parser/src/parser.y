%{
#include <cstdio>
#include <memory>
#include <vector>
#include "parser/node.hpp"
%}

%require "3.8"
%language "c++"
%define api.value.type variant
%define api.token.constructor
%define parse.error detailed
%locations

%code requires {
#include <memory>
#include <vector>
#include <string>
// Forward declarations for node types
class NBlock;
class NExpression;
class NStatement;
class NIdentifier;
class NQualifiedName;
class NVariableDeclaration;
class NFunctionDeclaration;
struct NLetBinding;
struct ImportItem;

// Type aliases (must match node.hpp)
using StatementList = std::vector<std::unique_ptr<NStatement>>;
using ExpressionList = std::vector<std::unique_ptr<NExpression>>;
using VariableList = std::vector<std::unique_ptr<NVariableDeclaration>>;
using StringList = std::vector<std::string>;
using LetBindingList = std::vector<std::unique_ptr<NLetBinding>>;
using ImportItemList = std::vector<ImportItem>;
}

%code provides {
// Global AST root (set after successful parse)
extern std::unique_ptr<NBlock> programBlock;
}

%code {
#include "parser/node.hpp"

// Global AST root (now uses unique_ptr for proper cleanup)
std::unique_ptr<NBlock> programBlock;

// Forward declaration of lexer function
yy::parser::symbol_type yylex();
}

// Tokens with string values
%token <std::string> TIDENTIFIER "identifier"
%token <std::string> TINTEGER "integer"
%token <std::string> TDOUBLE "double"

// Tokens without values
%token TCEQ "=="
%token TCNE "!="
%token TCLT "<"
%token TCLE "<="
%token TCGT ">"
%token TCGE ">="
%token TEQUAL "="
%token TLPAREN "("
%token TRPAREN ")"
%token TLBRACE "{"
%token TRBRACE "}"
%token TCOMMA ","
%token TDOT "."
%token TPLUS "+"
%token TMINUS "-"
%token TMUL "*"
%token TDIV "/"
%token TLET "let"
%token TFUN "fun"
%token TIN "in"
%token TCOLON ":"
%token TARROW "->"
%token TAND "and"
%token TIF "if"
%token TTHEN "then"
%token TELSE "else"
%token TTRUE "true"
%token TFALSE "false"
%token TMUT "mut"
%token TREF "ref"
%token TLARROW "<-"
%token TMODULE "module"
%token TENDMODULE "endmodule"
%token TIMPORT "import"
%token TFROM "from"
%token TAS "as"

// Nonterminal types with smart pointers
%type <std::unique_ptr<NIdentifier>> ident type_spec
%type <std::unique_ptr<NExpression>> numeric expr boolean
%type <std::unique_ptr<NBlock>> program stmts
%type <std::unique_ptr<NStatement>> stmt var_decl func_decl module_decl import_stmt
%type <std::unique_ptr<NVariableDeclaration>> func_param
%type <std::unique_ptr<NLetBinding>> let_binding
%type <std::unique_ptr<NQualifiedName>> qualified_name

// Vector types (by value, not pointer)
%type <VariableList> func_decl_args func_param_list
%type <ExpressionList> call_args
%type <LetBindingList> let_bindings
%type <StatementList> module_body
%type <StringList> ident_list
%type <ImportItemList> import_items

// Comparison operator type
%type <int> comparison

// Operator precedence (lowest to highest)
%right TLET TIN TAND
%right TIF TTHEN TELSE
%right TEQUAL TLARROW
%nonassoc COMPARISON TCEQ TCNE TCLT TCLE TCGT TCGE
%left TPLUS TMINUS
%left TMUL TDIV
%left TAS
%left TDOT
%precedence TREF         // ref expr (unary)
%precedence TDEREF       // *expr (unary dereference)

/* Expected conflicts:
   - ident TLPAREN (function call vs expr + (expr))
   - ident TDOT (qualified name vs expr DOT)
   - TMUL expr (unary deref vs binary multiplication)
   - TMUT expr in var_decl/let_binding with binary operators
*/
%expect 6

%start program

%%

program : stmts { programBlock = std::move($1); }
        ;

stmts : %empty { $$ = std::make_unique<NBlock>(); }
      | stmts stmt { $1->statements.push_back(std::move($2)); $$ = std::move($1); }
      ;

stmt : var_decl { $$ = std::move($1); }
     | func_decl { $$ = std::move($1); }
     | module_decl { $$ = std::move($1); }
     | import_stmt { $$ = std::move($1); }
     | expr { $$ = std::make_unique<NExpressionStatement>(std::move($1)); }
     ;

var_decl : TLET ident TEQUAL expr {
             /* let x = expr (immutable, type to be inferred) */
             $$ = std::make_unique<NVariableDeclaration>(std::move($2), std::move($4));
           }
         | TLET ident TCOLON type_spec TEQUAL expr {
             /* let x : type = expr (immutable) */
             $$ = std::make_unique<NVariableDeclaration>(std::move($4), std::move($2), std::move($6));
           }
         | TLET ident TEQUAL TMUT expr {
             /* let x = mut expr (mutable, type to be inferred from NMutRefExpression) */
             $$ = std::make_unique<NVariableDeclaration>(std::move($2),
                 std::make_unique<NMutRefExpression>(std::move($5)));
           }
         | TLET ident TCOLON type_spec TEQUAL TMUT expr {
             /* let x : type = mut expr (mutable) */
             $$ = std::make_unique<NVariableDeclaration>(std::move($4), std::move($2),
                 std::make_unique<NMutRefExpression>(std::move($7)));
           }
         ;

func_decl : TLET ident func_decl_args TCOLON type_spec TEQUAL expr {
              /* let fname (x : type) ... : rettype = expr */
              auto body = std::make_unique<NBlock>();
              body->statements.push_back(std::make_unique<NExpressionStatement>(std::move($7)));
              $$ = std::make_unique<NFunctionDeclaration>(std::move($5), std::move($2), std::move($3), std::move(body));
            }
          | TLET ident func_decl_args TEQUAL expr {
              /* let fname (x : type) ... = expr (return type to be inferred) */
              auto body = std::make_unique<NBlock>();
              body->statements.push_back(std::make_unique<NExpressionStatement>(std::move($5)));
              $$ = std::make_unique<NFunctionDeclaration>(std::move($2), std::move($3), std::move(body));
            }
          ;

func_decl_args : TLPAREN func_param_list TRPAREN {
              /* (x: type, y: type, ...) */
              $$ = std::move($2);
            }
          | TLPAREN TRPAREN {
              /* Empty parameter list: () */
              $$ = VariableList();
            }
          ;

func_param_list : func_param {
                /* First parameter */
                $$ = VariableList();
                $$.push_back(std::move($1));
              }
            | func_param_list TCOMMA func_param {
                /* Additional parameters */
                $1.push_back(std::move($3));
                $$ = std::move($1);
              }
            ;

func_param : ident TCOLON type_spec {
               /* x : type (explicit type annotation) */
               $$ = std::make_unique<NVariableDeclaration>(std::move($3), std::move($1), nullptr);
             }
           | ident {
               /* x (type to be inferred) */
               $$ = std::make_unique<NVariableDeclaration>(std::move($1), nullptr);
             }
           ;

/* Module declarations with Haskell-style export list */
module_decl : TMODULE ident TLPAREN ident_list TRPAREN module_body TENDMODULE {
                /* module Name (export1, export2, ...) ... endmodule */
                $$ = std::make_unique<NModuleDeclaration>(std::move($2), std::move($4), std::move($6));
              }
            | TMODULE ident module_body TENDMODULE {
                /* module Name ... endmodule (no exports, all private) */
                $$ = std::make_unique<NModuleDeclaration>(std::move($2), std::move($3));
              }
            ;

module_body : %empty { $$ = StatementList(); }
            | module_body var_decl { $1.push_back(std::move($2)); $$ = std::move($1); }
            | module_body func_decl { $1.push_back(std::move($2)); $$ = std::move($1); }
            | module_body module_decl { $1.push_back(std::move($2)); $$ = std::move($1); }
            ;

ident_list : ident {
               $$ = StringList();
               $$.push_back($1->name);
             }
           | ident_list TCOMMA ident {
               $1.push_back($3->name);
               $$ = std::move($1);
             }
           ;

/* Import statements */
import_stmt : TIMPORT qualified_name {
                /* import Math */
                $$ = std::make_unique<NImportStatement>(std::move($2));
              }
            | TIMPORT qualified_name TAS ident {
                /* import Math as M */
                $$ = std::make_unique<NImportStatement>(std::move($2), $4->name);
              }
            | TFROM qualified_name TIMPORT import_items {
                /* from Math import add, PI */
                $$ = std::make_unique<NImportStatement>(std::move($2), std::move($4), false);
              }
            | TFROM qualified_name TIMPORT TMUL {
                /* from Math import * */
                $$ = std::make_unique<NImportStatement>(std::move($2), ImportItemList(), true);
              }
            ;

qualified_name : ident {
                   StringList parts;
                   parts.push_back($1->name);
                   $$ = std::make_unique<NQualifiedName>(std::move(parts));
                 }
               | qualified_name TDOT ident {
                   $1->parts.push_back($3->name);
                   $$ = std::move($1);
                 }
               ;

import_items : ident {
                 $$ = ImportItemList();
                 $$.push_back(ImportItem($1->name));
               }
             | ident TAS ident {
                 $$ = ImportItemList();
                 $$.push_back(ImportItem($1->name, $3->name));
               }
             | import_items TCOMMA ident {
                 $1.push_back(ImportItem($3->name));
                 $$ = std::move($1);
               }
             | import_items TCOMMA ident TAS ident {
                 $1.push_back(ImportItem($3->name, $5->name));
                 $$ = std::move($1);
               }
             ;

ident : TIDENTIFIER { $$ = std::make_unique<NIdentifier>($1); }
      ;

type_spec : ident { $$ = std::move($1); }
          | TREF ident {
              $$ = std::make_unique<NIdentifier>("ref " + $2->name);
            }
          | TMUT ident {
              $$ = std::make_unique<NIdentifier>("mut " + $2->name);
            }
          ;

numeric : TINTEGER { $$ = std::make_unique<NInteger>(atol($1.c_str())); }
        | TDOUBLE { $$ = std::make_unique<NDouble>(atof($1.c_str())); }
        ;

boolean : TTRUE { $$ = std::make_unique<NBoolean>(true); }
        | TFALSE { $$ = std::make_unique<NBoolean>(false); }
        ;

expr : ident TLARROW expr { $$ = std::make_unique<NAssignment>(std::move($1), std::move($3)); }
     | ident TLPAREN call_args TRPAREN {
         $$ = std::make_unique<NMethodCall>(std::move($1), std::move($3));
       }
     | ident { $$ = std::move($1); }
     | ident TDOT ident TLPAREN call_args TRPAREN {
         /* Qualified function call: Math.add(1, 2) */
         StringList parts;
         parts.push_back($1->name);
         parts.push_back($3->name);
         auto qname = std::make_unique<NQualifiedName>(std::move(parts));
         $$ = std::make_unique<NMethodCall>(std::move(qname), std::move($5));
       }
     | ident TDOT ident TDOT ident TLPAREN call_args TRPAREN {
         /* Nested qualified function call: Math.Internal.helper(5) */
         StringList parts;
         parts.push_back($1->name);
         parts.push_back($3->name);
         parts.push_back($5->name);
         auto qname = std::make_unique<NQualifiedName>(std::move(parts));
         $$ = std::make_unique<NMethodCall>(std::move(qname), std::move($7));
       }
     | ident TDOT ident {
         /* Qualified variable access: Math.PI */
         StringList parts;
         parts.push_back($1->name);
         parts.push_back($3->name);
         $$ = std::make_unique<NQualifiedName>(std::move(parts));
       }
     | ident TDOT ident TDOT ident {
         /* Nested qualified variable access: Math.Internal.PI */
         StringList parts;
         parts.push_back($1->name);
         parts.push_back($3->name);
         parts.push_back($5->name);
         $$ = std::make_unique<NQualifiedName>(std::move(parts));
       }
     | numeric { $$ = std::move($1); }
     | boolean { $$ = std::move($1); }
     | expr comparison expr %prec COMPARISON {
         $$ = std::make_unique<NBinaryOperator>(std::move($1), $2, std::move($3));
       }
     | expr TPLUS expr {
         $$ = std::make_unique<NBinaryOperator>(std::move($1), yy::parser::token::TPLUS, std::move($3));
       }
     | expr TMINUS expr {
         $$ = std::make_unique<NBinaryOperator>(std::move($1), yy::parser::token::TMINUS, std::move($3));
       }
     | expr TMUL expr {
         $$ = std::make_unique<NBinaryOperator>(std::move($1), yy::parser::token::TMUL, std::move($3));
       }
     | expr TDIV expr {
         $$ = std::make_unique<NBinaryOperator>(std::move($1), yy::parser::token::TDIV, std::move($3));
       }
     | expr TAS ident {
         $$ = std::make_unique<NCastExpression>(std::move($1), std::move($3));
       }
     | TLPAREN expr TRPAREN { $$ = std::move($2); }
     | TIF expr TTHEN expr TELSE expr {
         $$ = std::make_unique<NIfExpression>(std::move($2), std::move($4), std::move($6));
       }
     | TLET let_bindings TIN expr {
         $$ = std::make_unique<NLetExpression>(std::move($2), std::move($4));
       }
     | TREF expr %prec TREF {
         /* ref expr - create immutable reference */
         $$ = std::make_unique<NRefExpression>(std::move($2));
       }
     | TMUL expr %prec TDEREF {
         /* *expr - dereference a reference */
         $$ = std::make_unique<NDerefExpression>(std::move($2));
       }
     ;

call_args : %empty { $$ = ExpressionList(); }
          | expr {
              $$ = ExpressionList();
              $$.push_back(std::move($1));
            }
          | call_args TCOMMA expr {
              $1.push_back(std::move($3));
              $$ = std::move($1);
            }
          ;

let_bindings : let_binding {
                 $$ = LetBindingList();
                 $$.push_back(std::move($1));
               }
             | let_bindings TAND let_binding {
                 $1.push_back(std::move($3));
                 $$ = std::move($1);
               }
             ;

let_binding : ident TEQUAL expr {
                /* x = expr (immutable, type to be inferred) */
                $$ = std::make_unique<NLetBinding>(
                    std::make_unique<NVariableDeclaration>(std::move($1), std::move($3)));
              }
            | ident TCOLON type_spec TEQUAL expr {
                /* x : type = expr (immutable) */
                $$ = std::make_unique<NLetBinding>(
                    std::make_unique<NVariableDeclaration>(std::move($3), std::move($1), std::move($5)));
              }
            | ident TEQUAL TMUT expr {
                /* x = mut expr (mutable, type to be inferred from NMutRefExpression) */
                $$ = std::make_unique<NLetBinding>(
                    std::make_unique<NVariableDeclaration>(std::move($1),
                        std::make_unique<NMutRefExpression>(std::move($4))));
              }
            | ident TCOLON type_spec TEQUAL TMUT expr {
                /* x : type = mut expr (mutable) */
                $$ = std::make_unique<NLetBinding>(
                    std::make_unique<NVariableDeclaration>(std::move($3), std::move($1),
                        std::make_unique<NMutRefExpression>(std::move($6))));
              }
            | ident func_decl_args TCOLON type_spec TEQUAL expr {
                /* f(x: int): int = expr */
                auto body = std::make_unique<NBlock>();
                body->statements.push_back(std::make_unique<NExpressionStatement>(std::move($6)));
                $$ = std::make_unique<NLetBinding>(
                    std::make_unique<NFunctionDeclaration>(std::move($4), std::move($1), std::move($2), std::move(body)));
              }
            | ident func_decl_args TEQUAL expr {
                /* f(x: int) = expr (return type inferred) */
                auto body = std::make_unique<NBlock>();
                body->statements.push_back(std::make_unique<NExpressionStatement>(std::move($4)));
                $$ = std::make_unique<NLetBinding>(
                    std::make_unique<NFunctionDeclaration>(std::move($1), std::move($2), std::move(body)));
              }
            ;

comparison : TCEQ { $$ = yy::parser::token::TCEQ; }
           | TCNE { $$ = yy::parser::token::TCNE; }
           | TCLT { $$ = yy::parser::token::TCLT; }
           | TCLE { $$ = yy::parser::token::TCLE; }
           | TCGT { $$ = yy::parser::token::TCGT; }
           | TCGE { $$ = yy::parser::token::TCGE; }
           ;

%%

namespace yy {
void parser::error(const location_type& loc, const std::string& msg) {
  // Use fprintf(stderr) to ensure error messages can be captured by tests
  // that redirect C's stderr stream (std::cerr is not affected by such redirects)
  fprintf(stderr, "ERROR: %s at line %d, column %d\n",
          msg.c_str(), loc.begin.line, loc.begin.column);
}
}
