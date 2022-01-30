## Social Force Model

This implements the Social Force Model. We adapted code from the Boids Flocker Model from from https://github.com/projectmesa/mesa/tree/main/examples/boid_flockers. However, most files have been severely changed and the file `boid.py` has been (severely) modified and renamed to `human.py`.
The model was adapted and tuned according to the equations and values from this paper: Helbing, D., Farkas, I. J., Molnar, P., & Vicsek, T. (2002). Simulation of pedestrian crowds in normal and evacuation situations. Pedestrian and evacuation dynamics, 21(2), 21-58.
We implemented the sensitivity analysis ourselves.

## How to run

Install requirements using `pip install -r requirements.txt`.

* Launch the visualization
```
    $ mesa runserver
```

Or:

```
    $ python server.py
```