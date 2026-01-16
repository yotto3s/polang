%{
#include <string>
#include "parser/node.hpp"
#include "parser.hpp"

// Forward declaration for unified error reporting
extern "C" void polang_report_error(const char* message, int line, int column);

#define SAVE_TOKEN yylval.string = new std::string(yytext, yyleng)
#define TOKEN(t) (yylval.token = t)

static int yycolumn = 1;

void polang_reset_lexer_location() {
    yycolumn = 1;
    yylineno = 1;
}

#define YY_USER_ACTION \
    yylloc.first_line = yylloc.last_line = yylineno; \
    yylloc.first_column = yycolumn; \
    yylloc.last_column = yycolumn + yyleng - 1; \
    yycolumn += yyleng;
%}

%option noyywrap
%option yylineno

%%

[ \t]+                  { /* skip spaces and tabs */ }
\n                      { yycolumn = 1; /* reset column on newline */ }
;[^\n]*                 { /* skip single-line comments */ }
"let"                   { return TOKEN(TLET); }
"fun"                   { return TOKEN(TFUN); }
"in"                    { return TOKEN(TIN); }
"and"                   { return TOKEN(TAND); }
"if"                    { return TOKEN(TIF); }
"then"                  { return TOKEN(TTHEN); }
"else"                  { return TOKEN(TELSE); }
"true"                  { return TOKEN(TTRUE); }
"false"                 { return TOKEN(TFALSE); }
"mut"                   { return TOKEN(TMUT); }
[a-zA-Z_][a-zA-Z0-9_]*  { SAVE_TOKEN; return TIDENTIFIER; }
[0-9]+\.[0-9]*          { SAVE_TOKEN; return TDOUBLE; }
[0-9]+                  { SAVE_TOKEN; return TINTEGER; }
"->"                    { return TOKEN(TARROW); }
"<-"                    { return TOKEN(TLARROW); }
"="                     { return TOKEN(TEQUAL); }
"=="                    { return TOKEN(TCEQ); }
"!="                    { return TOKEN(TCNE); }
"<"                     { return TOKEN(TCLT); }
"<="                    { return TOKEN(TCLE); }
">"                     { return TOKEN(TCGT); }
">="                    { return TOKEN(TCGE); }
"("                     { return TOKEN(TLPAREN); }
")"                     { return TOKEN(TRPAREN); }
"{"                     { return TOKEN(TLBRACE); }
"}"                     { return TOKEN(TRBRACE); }
"."                     { return TOKEN(TDOT); }
","                     { return TOKEN(TCOMMA); }
":"                     { return TOKEN(TCOLON); }
"+"                     { return TOKEN(TPLUS); }
"-"                     { return TOKEN(TMINUS); }
"*"                     { return TOKEN(TMUL); }
"/"                     { return TOKEN(TDIV); }
.                       { polang_report_error(yytext, yylineno, yylloc.first_column); yyterminate(); }

%%
