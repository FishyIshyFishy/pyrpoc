# pyrpoc themes

Themes are JSON files mapping color tokens to values. At startup (and whenever
the user picks a theme from **Style** in the menubar), `ThemeController`
renders `templates/stylesheet.qss.in` and the SVG icon templates in
`templates/icons/` by substituting each `^token^` placeholder with the theme's
color, then applies the result to the whole application. Nothing is
pregenerated: any color scheme works at runtime.

## Built-in themes

`themes/*.json` ships dark and light variants in six accent colors. These were
recovered from the BreezeStyleSheets-derived stylesheets that used to be
compiled into Qt resource blobs (`breeze_all.py`, removed): all non-alt themes
came from one template, so aligning the color literals across the 12 generated
stylesheets recovers both the shared template and each theme's token values.
That alignment was verified byte-for-byte before the blobs were deleted.

## Custom themes

Put a `my-theme.json` in the user themes folder (**Style → Open Themes
Folder...**, stored under the application data directory) and pick
**Style → Reload Themes**. A user theme with the same name as a built-in
overrides it.

A theme can inherit everything from another theme and override only a few
tokens:

```json
// lines starting with // are comments
{
    "base": "dark-pink",
    "highlight": "#39bae6",
    "highlight:dark": "#44a0d0",
    "highlight:alternate": "#3d8abb"
}
```

Without `"base"`, the file must define every token used by the templates
(copy a file from `themes/` as a starting point). Values are anything QSS/SVG
accepts where the token appears: `#rrggbb` hex or `rgba(r, g, b, a)`. The
`icon:*:opacity` tokens are plain numbers between 0 and 1.

The main accent tokens, if you just want "dark-pink but a different color":
`highlight`, `highlight:dark`, `highlight:alternate`, `slider:foreground`,
`scrollbar:hover`, `checkbox:light`, `view:checked`, `view:hover`,
`icon:hover`, `icon:pressed`, `ads-tab:focused`, `ads-border:focused`.
