# Blockman
Mark/Highlight code blocks

# Setup
```
// settings.json (User/Global config, not Workspace config)
// To open this file in VS Code, press F1, type 'settings json' and choose 'Preferences: Open Settings (JSON)'
{
    // ...
    "editor.inlayHints.enabled": "off",
    "editor.guides.indentation": false, // new API for indent guides. The old one is: "editor.renderIndentGuides": false,
    "editor.guides.bracketPairs": false, // advanced indent guides (But only for brackets) (This does not turn off editor.bracketPairColorization)
    // 自动换行
    "editor.wordWrap": "off",
    "diffEditor.wordWrap": "off",

    "workbench.colorCustomizations": {
        // ...
        "editor.lineHighlightBorder": "#9fced11f",
        "editor.lineHighlightBackground": "#1073cf2d"
    }
}
```

# Configuration
## Super gradients
## Support Double Width Chars
Press `F1` and type the command name: `Blockman Toggle Try Support Double Width Chars`.
