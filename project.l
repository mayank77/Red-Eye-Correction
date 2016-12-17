%{
#include<stdio.h>
%}
nl [\n]+
tb [\t]+
spc [ ]+
relop >|<|<=|>=|!=|==
ariop [+*\-\/]=
incdec [+][+]|[-][-]
dtaty void|int|float|char
cmt "//"[a-zA-Z ]*|"/*"[A-Za-z0-9 *\t\n.]*"*/"
var [a-zA-Z][a-zA-Z0-9]*
cnst [\-]?[0-9]+[\.]?[0-9]*
usrdtaty struct|union|enum
fun main|printf|scanf|strlen|strcpy
head "#include<"[a-zA-z]*".h>"
ignore [\[{\]}()"",&;]
string [\"][A-Za-z0-9 *\t\n.]*[\"]



%%
{nl} {;}
{tb} {;}
{ignore} {;}
{head} {fprintf(yyout,"%s\theader file\n",yytext);}
{ariop} {fprintf(yyout,"%s\t\tarithmetic operator\n",yytext);}
{relop} {fprintf(yyout,"%s\t\trelational operator\n",yytext);}
{fun} {fprintf(yyout,"%s\t\tfuntions\n",yytext);}
{incdec} {fprintf(yyout,"%s\t\tincrement/decreement operator\n",yytext);}
{dtaty} {fprintf(yyout,"%s\t\tdatatype\n",yytext);}
"=" {fprintf(yyout,"%s\t\tasignment operator\n",yytext);}
{usrdtaty} {fprintf(yyout,"%s\t\tuserdefined datatype\n",yytext);}
"if" {fprintf(yyout,"%s\t\tif keyword\n",yytext);}
"else if" {fprintf(yyout,"%s\t\telse if keyword\n",yytext);}
"else" {fprintf(yyout,"%s\t\telse keyword\n",yytext);}
"switch" {fprintf(yyout,"%s\t\tswitch keyword\n",yytext);}
"case:" {fprintf(yyout,"%s\t\tcase keyword\n",yytext);}
{string} {fprintf(yyout,"%s\t\tString\n",yytext);}
{cmt} {fprintf(yyout,"%s\tcomment\n",yytext);}
{var} {fprintf(yyout,"%s\t\tvariable\n",yytext);}
{cnst} {fprintf(yyout,"%s\t\tconstants\n",yytext);}
{spc} {;}

. {;}
%%
main()
{
yyin=fopen("abc.c","r");
yyout=fopen("xyz.txt","w");
yylex();
fclose(yyin);
fclose(yyout);
}

