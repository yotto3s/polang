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
#include "parser/type_var_decl.hpp"
// Forward declarations for node types
class NBlock;
class NExpression;
class NStatement;
class NIdentifier;
class NTypeSpec;
class NNamedType;
class NArrowType;
class NProductType;
class NTypeVar;
class NForallType;
class NUnitType;
class NUnitLiteral;
class NQualifiedName;
class NVariableDeclaration;
class NFunctionDeclaration;
class NTypeSignature;
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

// Macro to set source location on a node
#define SET_LOC(node, bisonLoc) \
  do { if (node) (node)->setLocation((bisonLoc).begin.line, (bisonLoc).begin.column); } while(0)

// Build a definition (variable or function) from LHS = RHS expressions.
// Returns nullptr if LHS is not a valid definition target.
static std::unique_ptr<NStatement>
buildDefinition(std::unique_ptr<NExpression> lhs,
                std::unique_ptr<NExpression> rhs) {
  auto* lhsIdent = dynamic_cast<NIdentifier*>(lhs.get());
  if (lhsIdent != nullptr) {
    auto id = std::unique_ptr<NIdentifier>(
        static_cast<NIdentifier*>(lhs.release()));
    return std::make_unique<NVariableDeclaration>(std::move(id), std::move(rhs));
  }
  auto* lhsCall = dynamic_cast<NMethodCall*>(lhs.get());
  if (lhsCall != nullptr && lhsCall->qualifiedId == nullptr) {
    VariableList params;
    for (auto& arg : lhsCall->arguments) {
      auto* argIdent = dynamic_cast<NIdentifier*>(arg.get());
      if (argIdent == nullptr) {
        return nullptr; // caller should report error
      }
      auto paramId = std::unique_ptr<NIdentifier>(
          static_cast<NIdentifier*>(arg.release()));
      params.push_back(std::make_unique<NVariableDeclaration>(
          std::move(paramId), nullptr));
    }
    auto funcId = std::unique_ptr<NIdentifier>(
        static_cast<NIdentifier*>(lhsCall->id.release()));
    auto body = std::make_unique<NBlock>();
    auto exprStmt = std::make_unique<NExpressionStatement>(std::move(rhs));
    body->statements.push_back(std::move(exprStmt));
    return std::make_unique<NFunctionDeclaration>(
        std::move(funcId), std::move(params), std::move(body));
  }
  return nullptr;
}

// Build a type signature from LHS : type_expr.
// Returns nullptr if LHS is not an identifier.
static std::unique_ptr<NStatement>
buildTypeSignature(std::unique_ptr<NExpression> lhs,
                   std::unique_ptr<const NTypeSpec> typeExpr) {
  auto* lhsIdent = dynamic_cast<NIdentifier*>(lhs.get());
  if (lhsIdent == nullptr) {
    return nullptr;
  }
  auto id = std::unique_ptr<NIdentifier>(
      static_cast<NIdentifier*>(lhs.release()));
  return std::make_unique<NTypeSignature>(std::move(id), std::move(typeExpr));
}
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
%token TMOD "%"
%token TLAND "&&"
%token TLOR "||"
%token TNOT "!"
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
%token TMODULE "module"
%token TENDMODULE "endmodule"
%token TIMPORT "import"
%token TFROM "from"
%token TAS "as"
%token TFORALL "forall"
%token <std::string> TTYPEVAR "typevar"

// Nonterminal types with smart pointers
%type <std::unique_ptr<NIdentifier>> ident
%type <std::unique_ptr<const NTypeSpec>> type_spec type_expr type_product type_atom
%type <std::vector<std::unique_ptr<const NTypeSpec>>> type_product_list
%type <std::vector<TypeVarDecl>> type_var_list
%type <TypeVarDecl> type_var_decl
%type <std::unique_ptr<NExpression>> numeric expr boolean
%type <std::unique_ptr<NBlock>> program stmts
%type <std::unique_ptr<NStatement>> stmt type_sig module_decl import_stmt
%type <std::unique_ptr<NVariableDeclaration>> func_param
%type <std::unique_ptr<NLetBinding>> let_binding
%type <std::unique_ptr<NQualifiedName>> qualified_name
%type <std::unique_ptr<NQualifiedName>> qualified_name_multi

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
%right TEQUAL
%left TLOR
%left TLAND
%nonassoc COMPARISON TCEQ TCNE TCLT TCLE TCGT TCGE
%left TPLUS TMINUS
%left TMUL TDIV TMOD
%left TAS
%right UNARY TNOT
%left TDOT

/* Expected shift/reduce conflicts (all on TLPAREN):
   1. expr . "(" in call_args (function call vs grouped expr)
   2. ident . "(" in module (export list vs module_body expr)
   3. ident . "(" in stmts (function call vs grouped expr)
   4. ident "." ident . "(" (qualified call vs qualified name + grouped expr)
   5. expr . "-" (binary minus vs start of unary negation after newline)
*/
%expect 5

%start program

%%

program : stmts { programBlock = std::move($1); }
        ;

stmts : %empty { $$ = std::make_unique<NBlock>(); SET_LOC($$, @$); }
      | stmts stmt { $1->statements.push_back(std::move($2)); $$ = std::move($1); }
      ;

stmt : expr TEQUAL expr {
         $$ = buildDefinition(std::move($1), std::move($3));
         if ($$ == nullptr) {
           error(@1, "invalid left-hand side of definition");
           YYERROR;
         }
         SET_LOC($$, @$);
       }
     | type_sig { $$ = std::move($1); }
     | module_decl { $$ = std::move($1); }
     | import_stmt { $$ = std::move($1); }
     | expr { $$ = std::make_unique<NExpressionStatement>(std::move($1)); SET_LOC($$, @$); }
     ;

type_sig : expr TCOLON type_expr {
             $$ = buildTypeSignature(std::move($1), std::move($3));
             if ($$ == nullptr) {
               error(@1, "type signature must be for an identifier");
               YYERROR;
             }
             SET_LOC($$, @$);
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
               SET_LOC($$, @$);
             }
           | ident {
               /* x (type to be inferred) */
               $$ = std::make_unique<NVariableDeclaration>(std::move($1), nullptr);
               SET_LOC($$, @$);
             }
           ;

/* Module declarations with Haskell-style export list */
module_decl : TMODULE ident TLPAREN ident_list TRPAREN module_body TENDMODULE {
                /* module Name (export1, export2, ...) ... endmodule */
                $$ = std::make_unique<NModuleDeclaration>(std::move($2), std::move($4), std::move($6));
                SET_LOC($$, @$);
              }
            | TMODULE ident module_body TENDMODULE {
                /* module Name ... endmodule (no exports, all private) */
                $$ = std::make_unique<NModuleDeclaration>(std::move($2), std::move($3));
                SET_LOC($$, @$);
              }
            ;

module_body : %empty { $$ = StatementList(); }
            | module_body expr TEQUAL expr {
                auto stmt = buildDefinition(std::move($2), std::move($4));
                if (stmt == nullptr) {
                  error(@2, "invalid definition in module");
                  YYERROR;
                }
                SET_LOC(stmt, @2);
                $1.push_back(std::move(stmt));
                $$ = std::move($1);
              }
            | module_body expr TCOLON type_expr {
                auto stmt = buildTypeSignature(std::move($2), std::move($4));
                if (stmt == nullptr) {
                  error(@2, "type signature must be for an identifier");
                  YYERROR;
                }
                SET_LOC(stmt, @2);
                $1.push_back(std::move(stmt));
                $$ = std::move($1);
              }
            | module_body module_decl {
                $1.push_back(std::move($2));
                $$ = std::move($1);
              }
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
                SET_LOC($$, @$);
              }
            | TIMPORT qualified_name TAS ident {
                /* import Math as M */
                $$ = std::make_unique<NImportStatement>(std::move($2), $4->name);
                SET_LOC($$, @$);
              }
            | TFROM qualified_name TIMPORT import_items {
                /* from Math import add, PI */
                $$ = std::make_unique<NImportStatement>(std::move($2), std::move($4), false);
                SET_LOC($$, @$);
              }
            | TFROM qualified_name TIMPORT TMUL {
                /* from Math import * */
                $$ = std::make_unique<NImportStatement>(std::move($2), ImportItemList(), true);
                SET_LOC($$, @$);
              }
            ;

qualified_name : ident {
                   StringList parts;
                   parts.push_back($1->name);
                   $$ = std::make_unique<NQualifiedName>(std::move(parts));
                   SET_LOC($$, @$);
                 }
               | qualified_name TDOT ident {
                   $1->parts.push_back($3->name);
                   $$ = std::move($1);
                   SET_LOC($$, @$);
                 }
               ;

/* Qualified name with at least 2 parts (guaranteed to have a dot) */
qualified_name_multi : ident TDOT ident {
                         StringList parts;
                         parts.push_back($1->name);
                         parts.push_back($3->name);
                         $$ = std::make_unique<NQualifiedName>(std::move(parts));
                         SET_LOC($$, @$);
                       }
                     | qualified_name_multi TDOT ident {
                         $1->parts.push_back($3->name);
                         $$ = std::move($1);
                         SET_LOC($$, @$);
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

ident : TIDENTIFIER { $$ = std::make_unique<NIdentifier>($1); SET_LOC($$, @$); }
      ;

type_spec : ident {
              $$ = std::make_unique<const NNamedType>($1->name);
            }
          ;

/* Type expressions for type signatures */
type_expr : TFORALL type_var_list TDOT type_expr {
              /* Forall quantifier: forall 'a, 'b:Numeric. 'a -> 'b */
              $$ = std::make_unique<const NForallType>(std::move($2), std::move($4));
            }
          | type_product TARROW type_expr {
              /* Function type: a -> b (right-associative) */
              $$ = std::make_unique<const NArrowType>(std::move($1), std::move($3));
            }
          | type_product {
              $$ = std::move($1);
            }
          ;

type_var_list : type_var_decl {
                  $$ = std::vector<TypeVarDecl>();
                  $$.push_back(std::move($1));
                }
              | type_var_list TCOMMA type_var_decl {
                  $1.push_back(std::move($3));
                  $$ = std::move($1);
                }
              ;

type_var_decl : TTYPEVAR {
                  /* 'a (unconstrained type variable) */
                  $$ = TypeVarDecl($1);
                }
              | TTYPEVAR TCOLON ident {
                  /* 'a:Numeric (constrained type variable) */
                  $$ = TypeVarDecl($1, $3->name);
                }
              ;

type_product : type_atom {
                 $$ = std::move($1);
               }
             | type_product_list {
                 $$ = std::make_unique<const NProductType>(std::move($1));
               }
             ;

type_product_list : type_atom TMUL type_atom {
                      $$ = std::vector<std::unique_ptr<const NTypeSpec>>();
                      $$.push_back(std::move($1));
                      $$.push_back(std::move($3));
                    }
                  | type_product_list TMUL type_atom {
                      $1.push_back(std::move($3));
                      $$ = std::move($1);
                    }
                  ;

type_atom : ident {
              $$ = std::make_unique<const NNamedType>($1->name);
            }
          | TTYPEVAR {
              /* Type variable reference: 'a */
              $$ = std::make_unique<const NTypeVar>($1);
            }
          | TLPAREN TRPAREN {
              /* () unit type */
              $$ = std::make_unique<const NUnitType>();
            }
          | TLPAREN type_expr TRPAREN {
              $$ = std::move($2);
            }
          ;

numeric : TINTEGER { $$ = std::make_unique<NInteger>(atol($1.c_str())); SET_LOC($$, @$); }
        | TDOUBLE { $$ = std::make_unique<NDouble>(atof($1.c_str())); SET_LOC($$, @$); }
        ;

boolean : TTRUE { $$ = std::make_unique<NBoolean>(true); SET_LOC($$, @$); }
        | TFALSE { $$ = std::make_unique<NBoolean>(false); SET_LOC($$, @$); }
        ;

expr : ident TLPAREN call_args TRPAREN {
         $$ = std::make_unique<NMethodCall>(std::move($1), std::move($3));
         SET_LOC($$, @$);
       }
     | ident { $$ = std::move($1); }
     | qualified_name_multi TLPAREN call_args TRPAREN {
         /* Qualified function call with arbitrary depth: Math.add(1, 2), A.B.C.func() */
         $$ = std::make_unique<NMethodCall>(std::move($1), std::move($3));
         SET_LOC($$, @$);
       }
     | qualified_name_multi {
         /* Qualified variable access with arbitrary depth: Math.PI, A.B.C.value */
         $$ = std::move($1);
       }
     | numeric { $$ = std::move($1); }
     | boolean { $$ = std::move($1); }
     | TMINUS expr %prec UNARY {
         $$ = std::make_unique<NUnaryOperator>(yy::parser::token::TMINUS, std::move($2));
         SET_LOC($$, @$);
       }
     | TNOT expr %prec UNARY {
         $$ = std::make_unique<NUnaryOperator>(yy::parser::token::TNOT, std::move($2));
         SET_LOC($$, @$);
       }
     | expr comparison expr %prec COMPARISON {
         $$ = std::make_unique<NBinaryOperator>(std::move($1), $2, std::move($3));
         SET_LOC($$, @$);
       }
     | expr TPLUS expr {
         $$ = std::make_unique<NBinaryOperator>(std::move($1), yy::parser::token::TPLUS, std::move($3));
         SET_LOC($$, @$);
       }
     | expr TMINUS expr {
         $$ = std::make_unique<NBinaryOperator>(std::move($1), yy::parser::token::TMINUS, std::move($3));
         SET_LOC($$, @$);
       }
     | expr TMUL expr {
         $$ = std::make_unique<NBinaryOperator>(std::move($1), yy::parser::token::TMUL, std::move($3));
         SET_LOC($$, @$);
       }
     | expr TDIV expr {
         $$ = std::make_unique<NBinaryOperator>(std::move($1), yy::parser::token::TDIV, std::move($3));
         SET_LOC($$, @$);
       }
     | expr TMOD expr {
         $$ = std::make_unique<NBinaryOperator>(std::move($1), yy::parser::token::TMOD, std::move($3));
         SET_LOC($$, @$);
       }
     | expr TLAND expr {
         $$ = std::make_unique<NBinaryOperator>(std::move($1), yy::parser::token::TLAND, std::move($3));
         SET_LOC($$, @$);
       }
     | expr TLOR expr {
         $$ = std::make_unique<NBinaryOperator>(std::move($1), yy::parser::token::TLOR, std::move($3));
         SET_LOC($$, @$);
       }
     | expr TAS type_spec {
         $$ = std::make_unique<NCastExpression>(std::move($1), std::move($3));
         SET_LOC($$, @$);
       }
     | TLPAREN TRPAREN {
         $$ = std::make_unique<NUnitLiteral>();
         SET_LOC($$, @$);
       }
     | TLPAREN expr TRPAREN { $$ = std::move($2); }
     | TIF expr TTHEN expr TELSE expr {
         $$ = std::make_unique<NIfExpression>(std::move($2), std::move($4), std::move($6));
         SET_LOC($$, @$);
       }
     | TLET let_bindings TIN expr {
         $$ = std::make_unique<NLetExpression>(std::move($2), std::move($4));
         SET_LOC($$, @$);
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
                auto varDecl = std::make_unique<NVariableDeclaration>(std::move($1), std::move($3));
                SET_LOC(varDecl, @$);
                $$ = std::make_unique<NLetBinding>(std::move(varDecl));
              }
            | ident TCOLON type_spec TEQUAL expr {
                /* x : type = expr (immutable) */
                auto varDecl = std::make_unique<NVariableDeclaration>(std::move($3), std::move($1), std::move($5));
                SET_LOC(varDecl, @$);
                $$ = std::make_unique<NLetBinding>(std::move(varDecl));
              }
            | ident func_decl_args TCOLON type_spec TEQUAL expr {
                /* f(x: int): int = expr */
                auto body = std::make_unique<NBlock>();
                SET_LOC(body, @6);
                auto exprStmt = std::make_unique<NExpressionStatement>(std::move($6));
                SET_LOC(exprStmt, @6);
                body->statements.push_back(std::move(exprStmt));
                auto funcDecl = std::make_unique<NFunctionDeclaration>(std::move($4), std::move($1), std::move($2), std::move(body));
                SET_LOC(funcDecl, @$);
                $$ = std::make_unique<NLetBinding>(std::move(funcDecl));
              }
            | ident func_decl_args TEQUAL expr {
                /* f(x: int) = expr (return type inferred) */
                auto body = std::make_unique<NBlock>();
                SET_LOC(body, @4);
                auto exprStmt = std::make_unique<NExpressionStatement>(std::move($4));
                SET_LOC(exprStmt, @4);
                body->statements.push_back(std::move(exprStmt));
                auto funcDecl = std::make_unique<NFunctionDeclaration>(std::move($1), std::move($2), std::move(body));
                SET_LOC(funcDecl, @$);
                $$ = std::make_unique<NLetBinding>(std::move(funcDecl));
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
