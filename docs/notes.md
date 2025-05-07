#  USEFUL COMMANDS
- to build in windows: colcon build --merge-install --packages-select pa_input_manager                               
- Service call from terminal: ros2 service call /generate_poses custom_interfaces/srv/GetInt ["{data: 0}"]          
- Install new packages for isaac sim python: ./python.sh -m pip install name_of_package_here                         

## git
- git status
- git checkout <name branch>
- git diff --name-only experiment
- git pull
- git branch
- git push origin --delete origin/revert-8c51726a
- git stash
- git checkout gu_branch
- git stash pop
- git status
- git add .
- git commit -m "First commit"
- git push
- git diff --name-only main
- git diff main src/prima_additiva/prima_additiva/simulation.py
- git merge main
- git checkout supsi --patch src/prima_additiva/prima_additiva/simulation
- git branch -d supsi
- git push -d origin NAMEBRANCH --- *to delete a branch in remote*

# Projects ideas

## Crows identifier
Is it possible to not just identify a crow, but also to identify which crow is which, based on some speciific thing? 
The result could be that the system can try to guess one crow e.g: Tim 50% Ronnie 50% whe the crow turn and we see a particular spot on its right wing
then it's 100% Ronnie. then we update all the previous frame not just of Ronnie but of Tim as well.

We use the combination of the two crows to better estimate the single. Just like us by exclusion can sometime classify something
