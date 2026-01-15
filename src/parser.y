%{
#include <vector>
#include "node.hpp"
NBlock *programBlock; /* the top level root node of our final AST */

extern int yylex();
void yyerror(const char *s) { printf("ERROR: %sn", s); }
%}

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
%token <token> TLET TFUN TIN TCOLON TARROW

/* Define the type of node our nonterminal symbols represent.
   The types refer to the %union declaration above. Ex: when
   we call an ident (defined by union type ident) we are really
   calling an (NIdentifier*). It makes the compiler happy.
 */
%type <ident> ident
%type <expr> numeric expr
%type <varvec> func_decl_args
%type <exprvec> call_args
%type <block> program stmts
%type <stmt> stmt var_decl func_decl
%type <token> comparison

/* Operator precedence (lowest to highest) */
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
        
stmts : stmt { $$ = new NBlock(); $$->statements.push_back($<stmt>1); }
      | stmts stmt { $1->statements.push_back($<stmt>2); }
      ;

stmt : var_decl | func_decl
     | expr { $$ = new NExpressionStatement(*$1); }
     ;

var_decl : TLET ident TEQUAL expr {
             /* let x = expr (type inferred as int for now) */
             NIdentifier *type = new NIdentifier("int");
             $$ = new NVariableDeclaration(*type, *$2, $4);
           }
         | TLET ident TCOLON ident TEQUAL expr {
             /* let x : type = expr */
             $$ = new NVariableDeclaration(*$4, *$2, $6);
           }
         ;
        
func_decl : TLET ident func_decl_args TCOLON ident TEQUAL expr {
              /* let fname (x : type) ... : rettype = expr */
              NBlock *body = new NBlock();
              body->statements.push_back(new NExpressionStatement(*$7));
              $$ = new NFunctionDeclaration(*$5, *$2, *$3, *body);
              delete $3;
            }
          | TLET ident func_decl_args TEQUAL expr {
              /* let fname (x : type) ... = expr (return type inferred) */
              NIdentifier *retType = new NIdentifier("int");
              NBlock *body = new NBlock();
              body->statements.push_back(new NExpressionStatement(*$5));
              $$ = new NFunctionDeclaration(*retType, *$2, *$3, *body);
              delete $3;
            }
          ;

func_decl_args : TLPAREN ident TCOLON ident TRPAREN {
              /* First argument: (x : type) */
              $$ = new VariableList();
              $$->push_back(new NVariableDeclaration(*$4, *$2));
            }
          | func_decl_args TLPAREN ident TCOLON ident TRPAREN {
              /* Additional arguments: (x : type) */
              $1->push_back(new NVariableDeclaration(*$5, *$3));
              $$ = $1;
            }
          ;

ident : TIDENTIFIER { $$ = new NIdentifier(*$1); delete $1; }
      ;

numeric : TINTEGER { $$ = new NInteger(atol($1->c_str())); delete $1; }
        | TDOUBLE { $$ = new NDouble(atof($1->c_str())); delete $1; }
        ;
    
expr : ident TEQUAL expr { $$ = new NAssignment(*$<ident>1, *$3); }
     | ident TLPAREN call_args TRPAREN { $$ = new NMethodCall(*$1, *$3); delete $3; }
     | ident { $<ident>$ = $1; }
     | numeric
     | expr comparison expr %prec COMPARISON { $$ = new NBinaryOperator(*$1, $2, *$3); }
     | expr TPLUS expr { $$ = new NBinaryOperator(*$1, TPLUS, *$3); }
     | expr TMINUS expr { $$ = new NBinaryOperator(*$1, TMINUS, *$3); }
     | expr TMUL expr { $$ = new NBinaryOperator(*$1, TMUL, *$3); }
     | expr TDIV expr { $$ = new NBinaryOperator(*$1, TDIV, *$3); }
     | TLPAREN expr TRPAREN { $$ = $2; }
     ;
    
call_args : /*blank*/  { $$ = new ExpressionList(); }
          | expr { $$ = new ExpressionList(); $$->push_back($1); }
          | call_args TCOMMA expr  { $1->push_back($3); }
          ;

comparison : TCEQ | TCNE | TCLT | TCLE | TCGT | TCGE
           ;

%%
