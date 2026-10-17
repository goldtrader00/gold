# Gold levels website: one-time setup (about 10 minutes, free)

1. Make a free account at github.com. Click New repository, name it `gold`, set it to **Public**, tick "Add a README", Create.
2. In the repo click Add file > Upload files. Drag in EVERYTHING from this folder, including the hidden `.github` folder (if Windows hides it, turn on "Hidden items" in File Explorer > View). Commit.
3. Settings > Pages > Source: "Deploy from a branch", Branch: main, Folder: /docs > Save.
4. Settings > Actions > General > Workflow permissions: "Read and write permissions" > Save.
5. Actions tab > "update-levels" > Run workflow. Wait about 1-2 minutes until it shows a green tick.
   (Red cross? Open the run and send me the log text.)
6. Your site: https://YOUR-USERNAME.github.io/gold/  (open it on the laptop, add to the phone home screen)

After that it updates itself every 15 minutes during US hours, Monday to Friday.
