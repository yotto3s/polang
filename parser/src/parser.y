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
    NVariableDeclaration *var_decl;
    std::vector<NVariableDeclaration*> *varvec;
    std::vector<NExpression*> *exprvec;
    NLetBinding *letbind;
    std::vector<NLetBinding*> *letbindvec;
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

/* Define the type of node our nonterminal symbols represent.
   The types refer to the %union declaration above. Ex: when
   we call an ident (defined by union type ident) we are really
   calling an (NIdentifier*). It makes the compiler happy.
 */
%type <ident> ident
%type <expr> numeric expr boolean
%type <varvec> func_decl_args func_param_list
%type <letbind> let_binding
%type <letbindvec> let_bindings
%type <exprvec> call_args
%type <block> program stmts
%type <stmt> stmt var_decl func_decl
%type <token> comparison

/* Operator precedence (lowest to highest) */
%right TLET TIN TAND
%right TIF TTHEN TELSE
%right TEQUAL
%nonassoc COMPARISON TCEQ TCNE TCLT TCLE TCGT TCGE
%left TPLUS TMINUS
%left TMUL TDIV

/* Expected conflicts: ident TLPAREN (function call vs expr + (expr)) */
%expect 1

%start program

%%

program : stmts { programBlock = $1; }
        ;
        
stmts : /* empty */ { $$ = new NBlock(); }
      | stmts stmt { $1->statements.push_back($<stmt>2); }
      ;

stmt : var_decl | func_decl
     | expr { $$ = new NExpressionStatement(*$1); }
     ;

var_decl : TLET ident TEQUAL expr {
             /* let x = expr (type to be inferred) */
             $$ = new NVariableDeclaration(*$2, $4);
           }
         | TLET ident TCOLON ident TEQUAL expr {
             /* let x : type = expr */
             $$ = new NVariableDeclaration($4, *$2, $6);
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

func_param_list : ident TCOLON ident {
                /* First parameter: x : type */
                $$ = new VariableList();
                $$->push_back(new NVariableDeclaration($3, *$1, nullptr));
              }
            | func_param_list TCOMMA ident TCOLON ident {
                /* Additional parameters: , x : type */
                $1->push_back(new NVariableDeclaration($5, *$3, nullptr));
                $$ = $1;
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

expr : ident TEQUAL expr { $$ = new NAssignment(*$<ident>1, *$3); }
     | ident TLPAREN call_args TRPAREN { $$ = new NMethodCall(*$1, *$3); delete $3; }
     | ident { $<ident>$ = $1; }
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
                /* x = expr (type to be inferred) */
                $$ = new NLetBinding(new NVariableDeclaration(*$1, $3));
              }
            | ident TCOLON ident TEQUAL expr {
                /* x : type = expr */
                $$ = new NLetBinding(new NVariableDeclaration($3, *$1, $5));
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
