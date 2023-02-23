# GistPad
Edit GitHub Gists and repositories.

# Gist Management

## create a new gist
Simply open up the Gists tree in the GistPad tab, and click the + icon in the toolbar and specify the description and files to seed it with (including support for directories!)

## create gists from local files or snippets
Right-clicking them in the Explorer tree, or right-clicking an editor window/tab, and selecting Copy File to Gist, Add Selection to Gist or Paste Gist File Contents.

Alternative, you can run the GistPad: New Gist and GistPad: New Secret Gist commands to create gists.

## Sorting and Grouping
alphabetically click the sort toggle button on the Gists tree's toolbar.

group them by type click the group toggle button on the Gists tree's toolbar.

## Gist Types
- note - Gists that are composed of nothing but .txt, .md/.markdown or .adoc files
- notebook - Gists that are compose of nothing by Jupyter Notebook files (.ipynb)
- code-swing - Gists that include either a codeswing.json file and/or an index.html file. Read more about swings here.
- code-swing-template - Swings whose codeswing.json file sets the template property to true. Read more about swing templates here.
- code-swing-tutorial - Swings whose codeswing.json file specifies a tutorial property. Read more about tutorials here.
- code-tour - Gists that include a main.tour file, and were created by exporting a CodeTour.
- diagram - Gists that include a .drawio file.
- flash-code - Gists that include a .deck file.
- code-snippet - Gists that don't match any of the above more-specific types.

if you want to group gists by your own custom types, then simply add a tag to the end of the gist's description, using the following format: #tag (or #tag-name)

## Files and Directories
if you create a new gist, and specify todos/personal.txt,todos/work.txt,reminders.txt, the gist will include a reminders.txt file at the root of the gist, and personal.txt and reminders.txt files within a new directory called todos.

## Pasting Images
right-clicking the editor and selecting Paste Image, or using one of the following keyboard shortcuts: ctrl + shift + v (Windows/Linux)

## GistLog
- run the GistPad: New GistLog command
- The blog.md file will be automatically opened for editing, 
- as soon as you're ready to publish your post, open gistlog.yml and set the published property to true. 
- Then, right-click your Gist and select the Open Gist in GistLog menu. 
- This will open your browser to the URL that you can share with others, in order to read your new post.

In addition to being able to view individual posts on GistLog, you can also open your entire feed by right-clicking the Your Gists tree node and selecting the Open Feed in GistLog menu item. This will launch your GistLog landing page that displays are published GistLog posts.
