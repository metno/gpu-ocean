# Design Pattern for the Implementation of the GPU Ocean Project

Classes:        upper CamelCase
Functions:      lower camelCase
Variables:      snake_case


# Workflow 

>   Distinction remotes when working with forks:
>   -   ``upstream``: metno remote of the root repository 
>   -   ``origin``: user remote of the own fork

-   Make sure to be on ``origin/master`` and update from metno remote
    ```
    git checkout master 
    git pull upstream master
    ```
-   Create branch for latest developments
    ```
    git checkout -b <new_dev_branch>
    ```
-   Stage, commit and push changes to the ``origin``
    ```
    git push -u origin HEAD 
    ```
-   Go to the ``github.com`` homepage and create a "pull request"
-   Either Martin or Håvard have to accept it...
-   Update your local master with changes in the metno remote by first fetching and then merging the ``upstream/master`` branch 
    ```
    git checkout master 
    git fetch upstream
    git merge upstream/master
    ```
    (Note that a usual ``pull`` would try to update from the own ``origin``)
