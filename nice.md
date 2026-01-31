Context:  
Hackathon. Theme: "Self-Improving Agents" w/WandB. 




Demo Goals:

What do we want to show?

Full loop, 1 task.
- t0: nothing
- t1: skill
- t2-t5: refinements
- t6: performance at conflict resolution

How?
- Weave: (DEFER UNTIL READY)
    - LLM as a judge result after each task
    - honestly maybe vibecode a UI and have weave on the side? idk
    - maybe for visualizing the traces ?

Want to demonstrate performance improvement . idk how. maybe the RLM trajs themselves ? hm... 

skill,creation, application, seeing performance after he skil has been refined.

though the cool shit is like ... the rlm using each traj to get better ! we should see how that works.

use fast model, bad at jj, it goes from zero to resolve the merge conflict. 

then, figure out how we can show the RLM using the outputs from its previous reflections to improve the instructions. 





====
Open problems
Environment/Execution:

- Visualize trajectories (vibecode): P3
- How do we put a skill in this environment that the RLM could use?: (completed: use <jj_info> block)
- Integrate RLM over trajectories: P0
        - Approach: 
            - Analyze the gold resolution (merge conflict from repo), intent behind the commits on both ends and what to preserve, and have the model check to see IF the intent was preserved (currently have oracle that sees if tests pass, but there are multiple ways it can be done).
            - no. this is feedback fn. what the RLM iterates on. 
                - 
            - We have a working impl in reference/archive for the LLM judge pattern. reference/archive is where to go for the setup, it's done well.
        - 

