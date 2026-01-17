%{
#include <vector>
#include "parser/node.hpp"
NBlock *programBlock; /* the top level root node of our final AST */

extern int yylex();
void yyerror(const char *s);
%}

%locations

/* Represents the many different ways we can access our data */
%union {
    Node *node;
    NBlock *block;
    NExpression *expr;
    NStatement *stmt;
    NIdentifier *ident;
    NQualifiedName *qualname;
    NVariableDeclaration *var_decl;
    std::vector<NVariableDeclaration*> *varvec;
    std::vector<NExpression*> *exprvec;
    NLetBinding *letbind;
    std::vector<NLetBinding*> *letbindvec;
    std::vector<std::string> *strvec;
    std::vector<NStatement*> *stmtvec;
    ImportItemList *importitems;
    std::string *string;
    int token;
}

/* Define our terminal symbols (tokens). This should
   match our tokens.l lex file. We also define the node type
   they represent.
 */
%token <string> TIDENTIFIER TINTEGER TDOUBLE
%token <token> TCEQ TCNE TCLT TCLE TCGT TCGE TEQUAL
%token <token> TLPAREN TRPAREN TLBRACE TRBRACE TCOMMA TDOT
%token <token> TPLUS TMINUS TMUL TDIV
%token <token> TLET TFUN TIN TCOLON TARROW TAND
%token <token> TIF TTHEN TELSE
%token <token> TTRUE TFALSE
%token <token> TMUT TLARROW
%token <token> TMODULE TENDMODULE
%token <token> TIMPORT TFROM TAS

/* Define the type of node our nonterminal symbols represent.
   The types refer to the %union declaration above. Ex: when
   we call an ident (defined by union type ident) we are really
   calling an (NIdentifier*). It makes the compiler happy.
 */
%type <ident> ident
%type <expr> numeric expr boolean
%type <varvec> func_decl_args func_param_list
%type <var_decl> func_param
%type <letbind> let_binding
%type <letbindvec> let_bindings
%type <exprvec> call_args
%type <block> program stmts
%type <stmt> stmt var_decl func_decl module_decl import_stmt
%type <stmtvec> module_body
%type <strvec> ident_list
%type <importitems> import_items
%type <qualname> qualified_name
%type <token> comparison

/* Operator precedence (lowest to highest) */
%right TLET TIN TAND
%right TIF TTHEN TELSE
%right TEQUAL TLARROW
%nonassoc COMPARISON TCEQ TCNE TCLT TCLE TCGT TCGE
%left TPLUS TMINUS
%left TMUL TDIV
%left TDOT

/* Expected conflicts:
   - ident TLPAREN (function call vs expr + (expr))
   - ident TDOT (qualified name vs expr DOT)
*/
%expect 3

%start program

%%

program : stmts { programBlock = $1; }
        ;
        
stmts : /* empty */ { $$ = new NBlock(); }
      | stmts stmt { $1->statements.push_back($<stmt>2); }
      ;

stmt : var_decl | func_decl | module_decl | import_stmt
     | expr { $$ = new NExpressionStatement(*$1); }
     ;

var_decl : TLET ident TEQUAL expr {
             /* let x = expr (immutable, type to be inferred) */
             $$ = new NVariableDeclaration(*$2, $4, false);
           }
         | TLET ident TCOLON ident TEQUAL expr {
             /* let x : type = expr (immutable) */
             $$ = new NVariableDeclaration($4, *$2, $6, false);
           }
         | TLET TMUT ident TEQUAL expr {
             /* let mut x = expr (mutable, type to be inferred) */
             $$ = new NVariableDeclaration(*$3, $5, true);
           }
         | TLET TMUT ident TCOLON ident TEQUAL expr {
             /* let mut x : type = expr (mutable) */
             $$ = new NVariableDeclaration($5, *$3, $7, true);
           }
         ;
        
func_decl : TLET ident func_decl_args TCOLON ident TEQUAL expr {
              /* let fname (x : type) ... : rettype = expr */
              NBlock *body = new NBlock();
              body->statements.push_back(new NExpressionStatement(*$7));
              $$ = new NFunctionDeclaration($5, *$2, *$3, *body);
              delete $3;
            }
          | TLET ident func_decl_args TEQUAL expr {
              /* let fname (x : type) ... = expr (return type to be inferred) */
              NBlock *body = new NBlock();
              body->statements.push_back(new NExpressionStatement(*$5));
              $$ = new NFunctionDeclaration(*$2, *$3, *body);
              delete $3;
            }
          ;

func_decl_args : TLPAREN func_param_list TRPAREN {
              /* (x: type, y: type, ...) */
              $$ = $2;
            }
          | TLPAREN TRPAREN {
              /* Empty parameter list: () */
              $$ = new VariableList();
            }
          ;

func_param_list : func_param {
                /* First parameter */
                $$ = new VariableList();
                $$->push_back($1);
              }
            | func_param_list TCOMMA func_param {
                /* Additional parameters */
                $1->push_back($3);
                $$ = $1;
              }
            ;

func_param : ident TCOLON ident {
               /* x : type (explicit type annotation) */
               $$ = new NVariableDeclaration($3, *$1, nullptr);
             }
           | ident {
               /* x (type to be inferred) */
               $$ = new NVariableDeclaration(*$1, nullptr, false);
             }
           ;

/* Module declarations with Haskell-style export list */
module_decl : TMODULE ident TLPAREN ident_list TRPAREN module_body TENDMODULE {
                /* module Name (export1, export2, ...) ... endmodule */
                $$ = new NModuleDeclaration(*$2, *$4, *$6);
                delete $4;
                delete $6;
              }
            | TMODULE ident module_body TENDMODULE {
                /* module Name ... endmodule (no exports, all private) */
                $$ = new NModuleDeclaration(*$2, *$3);
                delete $3;
              }
            ;

module_body : /* empty */ { $$ = new StatementList(); }
            | module_body var_decl { $1->push_back($2); }
            | module_body func_decl { $1->push_back($2); }
            | module_body module_decl { $1->push_back($2); }
            ;

ident_list : ident {
               $$ = new StringList();
               $$->push_back($1->name);
               delete $1;
             }
           | ident_list TCOMMA ident {
               $1->push_back($3->name);
               delete $3;
             }
           ;

/* Import statements */
import_stmt : TIMPORT qualified_name {
                /* import Math */
                $$ = new NImportStatement(*$2);
              }
            | TIMPORT qualified_name TAS ident {
                /* import Math as M */
                $$ = new NImportStatement(*$2, $4->name);
                delete $4;
              }
            | TFROM qualified_name TIMPORT import_items {
                /* from Math import add, PI */
                $$ = new NImportStatement(*$2, *$4, false);
                delete $4;
              }
            | TFROM qualified_name TIMPORT TMUL {
                /* from Math import * */
                $$ = new NImportStatement(*$2, ImportItemList(), true);
              }
            ;

qualified_name : ident {
                   StringList parts;
                   parts.push_back($1->name);
                   $$ = new NQualifiedName(parts);
                   delete $1;
                 }
               | qualified_name TDOT ident {
                   $1->parts.push_back($3->name);
                   $$ = $1;
                   delete $3;
                 }
               ;

import_items : ident {
                 $$ = new ImportItemList();
                 $$->push_back(ImportItem($1->name));
                 delete $1;
               }
             | ident TAS ident {
                 $$ = new ImportItemList();
                 $$->push_back(ImportItem($1->name, $3->name));
                 delete $1;
                 delete $3;
               }
             | import_items TCOMMA ident {
                 $1->push_back(ImportItem($3->name));
                 delete $3;
               }
             | import_items TCOMMA ident TAS ident {
                 $1->push_back(ImportItem($3->name, $5->name));
                 delete $3;
                 delete $5;
               }
             ;

ident : TIDENTIFIER { $$ = new NIdentifier(*$1); delete $1; }
      ;

numeric : TINTEGER { $$ = new NInteger(atol($1->c_str())); delete $1; }
        | TDOUBLE { $$ = new NDouble(atof($1->c_str())); delete $1; }
        ;

boolean : TTRUE { $$ = new NBoolean(true); }
        | TFALSE { $$ = new NBoolean(false); }
        ;

expr : ident TLARROW expr { $$ = new NAssignment(*$<ident>1, *$3); }
     | ident TLPAREN call_args TRPAREN { $$ = new NMethodCall(*$1, *$3); delete $3; }
     | ident { $<ident>$ = $1; }
     | ident TDOT ident TLPAREN call_args TRPAREN {
         /* Qualified function call: Math.add(1, 2) */
         StringList parts;
         parts.push_back($1->name);
         parts.push_back($3->name);
         NQualifiedName *qname = new NQualifiedName(parts);
         $$ = new NMethodCall(*qname, *$5);
         delete $5;
       }
     | ident TDOT ident TDOT ident TLPAREN call_args TRPAREN {
         /* Nested qualified function call: Math.Internal.helper(5) */
         StringList parts;
         parts.push_back($1->name);
         parts.push_back($3->name);
         parts.push_back($5->name);
         NQualifiedName *qname = new NQualifiedName(parts);
         $$ = new NMethodCall(*qname, *$7);
         delete $7;
       }
     | ident TDOT ident {
         /* Qualified variable access: Math.PI */
         StringList parts;
         parts.push_back($1->name);
         parts.push_back($3->name);
         $$ = new NQualifiedName(parts);
       }
     | ident TDOT ident TDOT ident {
         /* Nested qualified variable access: Math.Internal.PI */
         StringList parts;
         parts.push_back($1->name);
         parts.push_back($3->name);
         parts.push_back($5->name);
         $$ = new NQualifiedName(parts);
       }
     | numeric
     | boolean
     | expr comparison expr %prec COMPARISON { $$ = new NBinaryOperator(*$1, $2, *$3); }
     | expr TPLUS expr { $$ = new NBinaryOperator(*$1, TPLUS, *$3); }
     | expr TMINUS expr { $$ = new NBinaryOperator(*$1, TMINUS, *$3); }
     | expr TMUL expr { $$ = new NBinaryOperator(*$1, TMUL, *$3); }
     | expr TDIV expr { $$ = new NBinaryOperator(*$1, TDIV, *$3); }
     | TLPAREN expr TRPAREN { $$ = $2; }
     | TIF expr TTHEN expr TELSE expr { $$ = new NIfExpression(*$2, *$4, *$6); }
     | TLET let_bindings TIN expr { $$ = new NLetExpression(*$2, *$4); delete $2; }
     ;
    
call_args : /*blank*/  { $$ = new ExpressionList(); }
          | expr { $$ = new ExpressionList(); $$->push_back($1); }
          | call_args TCOMMA expr  { $1->push_back($3); }
          ;

let_bindings : let_binding {
                 $$ = new LetBindingList();
                 $$->push_back($1);
               }
             | let_bindings TAND let_binding {
                 $1->push_back($3);
                 $$ = $1;
               }
             ;

let_binding : ident TEQUAL expr {
                /* x = expr (immutable, type to be inferred) */
                $$ = new NLetBinding(new NVariableDeclaration(*$1, $3, false));
              }
            | ident TCOLON ident TEQUAL expr {
                /* x : type = expr (immutable) */
                $$ = new NLetBinding(new NVariableDeclaration($3, *$1, $5, false));
              }
            | TMUT ident TEQUAL expr {
                /* mut x = expr (mutable, type to be inferred) */
                $$ = new NLetBinding(new NVariableDeclaration(*$2, $4, true));
              }
            | TMUT ident TCOLON ident TEQUAL expr {
                /* mut x : type = expr (mutable) */
                $$ = new NLetBinding(new NVariableDeclaration($4, *$2, $6, true));
              }
            | ident func_decl_args TCOLON ident TEQUAL expr {
                /* f(x: int): int = expr */
                NBlock *body = new NBlock();
                body->statements.push_back(new NExpressionStatement(*$6));
                $$ = new NLetBinding(new NFunctionDeclaration($4, *$1, *$2, *body));
                delete $2;
              }
            | ident func_decl_args TEQUAL expr {
                /* f(x: int) = expr (return type inferred) */
                NBlock *body = new NBlock();
                body->statements.push_back(new NExpressionStatement(*$4));
                $$ = new NLetBinding(new NFunctionDeclaration(*$1, *$2, *body));
                delete $2;
              }
            ;

comparison : TCEQ | TCNE | TCLT | TCLE | TCGT | TCGE
           ;

%%

void yyerror(const char *s) {
  fprintf(stderr, "ERROR: %s at line %d, column %d\n",
          s, yylloc.first_line, yylloc.first_column);
}
