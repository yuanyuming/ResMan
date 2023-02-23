# EditorConfig
Helps maintain consistent coding styles for multiple developers working on the same project across various editors and IDEs.

# Ussage
Right-clicking in the folder where you'd like it to be and selecting Generate .editorconfig.

# Example file
```
# EditorConfig is awesome: https://EditorConfig.org

# top-most EditorConfig file
root = true

# Unix-style newlines with a newline ending every file
[*]
end_of_line = lf
insert_final_newline = true

# Matches multiple files with brace expansion notation
# Set default charset
[*.{js,py}]
charset = utf-8

# 4 space indentation
[*.py]
indent_style = space
indent_size = 4

# Tab indentation (no size specified)
[Makefile]
indent_style = tab

# Indentation override for all JS under lib directory
[lib/**.js]
indent_style = space
indent_size = 2

# Matches the exact files either package.json or .travis.yml
[{package.json,.travis.yml}]
indent_style = space
indent_size = 2
```
# File Format Details
## Wildcard Patterns
*	Matches any string of characters, except path separators (/)
**	Matches any string of characters
?	Matches any single character
[name]	Matches any single character in name
[!name]	Matches any single character not in name
{s1,s2,s3}	Matches any of the strings given (separated by commas) 
{num1..num2}	Matches any integer numbers between num1 and num2, where num1 and num2 can be either positive or negative
## Supported Properties
- indent_style: set to tab or space to use hard tabs or soft tabs respectively.
- indent_size: a whole number defining the number of columns used for each indentation level and the width of soft tabs (when supported). When set to tab, the value of tab_width (if specified) will be used.
- tab_width: a whole number defining the number of columns used to represent a tab character. This defaults to the value of indent_size and doesn't usually need to be specified.
- end_of_line: set to lf, cr, or crlf to control how line breaks are represented.
- charset: set to latin1, utf-8, utf-8-bom, utf-16be or utf-16le to control the character set.
- trim_trailing_whitespace: set to true to remove any whitespace characters preceding newline characters and false to ensure it doesn't.
- insert_final_newline: set to true to ensure file ends with a newline when saving and false to ensure it doesn't.
- root: special property that should be specified at the top of the file outside of any sections. Set to true to stop .editorconfig files search on current file.
# Links
- [editorconfig.org](https://editorconfig.org/)
- [Projects Using EditorConfig](https://github.com/editorconfig/editorconfig/wiki/Projects-Using-EditorConfig)
- [Specification](https://github.com/editorconfig/editorconfig/wiki/Projects-Using-EditorConfig)
- [Properties](https://github.com/editorconfig/editorconfig/wiki/EditorConfig-Properties)