%{
#include <string>
#include "parser/node.hpp"
#include "parser.hpp"

// Forward declaration for unified error reporting
extern "C" void polang_report_error(const char* message, int line, int column);

// Global location variable for the C++ parser
yy::parser::location_type yylloc;

static int yycolumn = 1;

void polang_reset_lexer_location() {
    yycolumn = 1;
    yylineno = 1;
    yylloc = yy::parser::location_type();
}

// Create location for token constructors
#define YY_USER_ACTION \
    yylloc.begin.line = yylloc.end.line = yylineno; \
    yylloc.begin.column = yycolumn; \
    yylloc.end.column = yycolumn + yyleng - 1; \
    yycolumn += yyleng;

// Convenience macro for location
#define LOC yylloc

// Override lexer function to return symbol_type for C++ parser
#define YY_DECL yy::parser::symbol_type yylex()
%}

%option noyywrap
%option yylineno

%%

[ \t]+                  { /* skip spaces and tabs */ }
\n                      { yycolumn = 1; /* newline resets column */ }
;[^\n]*                 { /* skip single-line comments */ }
"let"                   { return yy::parser::make_TLET(LOC); }
"fun"                   { return yy::parser::make_TFUN(LOC); }
"in"                    { return yy::parser::make_TIN(LOC); }
"and"                   { return yy::parser::make_TAND(LOC); }
"if"                    { return yy::parser::make_TIF(LOC); }
"then"                  { return yy::parser::make_TTHEN(LOC); }
"else"                  { return yy::parser::make_TELSE(LOC); }
"true"                  { return yy::parser::make_TTRUE(LOC); }
"false"                 { return yy::parser::make_TFALSE(LOC); }
"mut"                   { return yy::parser::make_TMUT(LOC); }
"ref"                   { return yy::parser::make_TREF(LOC); }
"module"                { return yy::parser::make_TMODULE(LOC); }
"endmodule"             { return yy::parser::make_TENDMODULE(LOC); }
"import"                { return yy::parser::make_TIMPORT(LOC); }
"from"                  { return yy::parser::make_TFROM(LOC); }
"as"                    { return yy::parser::make_TAS(LOC); }
[a-zA-Z_][a-zA-Z0-9_]*  { return yy::parser::make_TIDENTIFIER(std::string(yytext, yyleng), LOC); }
[0-9]+\.[0-9]*          { return yy::parser::make_TDOUBLE(std::string(yytext, yyleng), LOC); }
[0-9]+                  { return yy::parser::make_TINTEGER(std::string(yytext, yyleng), LOC); }
"->"                    { return yy::parser::make_TARROW(LOC); }
"<-"                    { return yy::parser::make_TLARROW(LOC); }
"="                     { return yy::parser::make_TEQUAL(LOC); }
"=="                    { return yy::parser::make_TCEQ(LOC); }
"!="                    { return yy::parser::make_TCNE(LOC); }
"<"                     { return yy::parser::make_TCLT(LOC); }
"<="                    { return yy::parser::make_TCLE(LOC); }
">"                     { return yy::parser::make_TCGT(LOC); }
">="                    { return yy::parser::make_TCGE(LOC); }
"("                     { return yy::parser::make_TLPAREN(LOC); }
")"                     { return yy::parser::make_TRPAREN(LOC); }
"{"                     { return yy::parser::make_TLBRACE(LOC); }
"}"                     { return yy::parser::make_TRBRACE(LOC); }
"."                     { return yy::parser::make_TDOT(LOC); }
","                     { return yy::parser::make_TCOMMA(LOC); }
":"                     { return yy::parser::make_TCOLON(LOC); }
"+"                     { return yy::parser::make_TPLUS(LOC); }
"-"                     { return yy::parser::make_TMINUS(LOC); }
"*"                     { return yy::parser::make_TMUL(LOC); }
"/"                     { return yy::parser::make_TDIV(LOC); }
<<EOF>>                 { return yy::parser::make_YYEOF(LOC); }
.                       { polang_report_error(yytext, yylineno, yylloc.begin.column); return yy::parser::make_YYEOF(LOC); }

%%
