# Simulation of the active Brownian motion of a microswimmer (Volpe, et al.)
## Introduction 
Microswimmers - active Brownian particles. Interplay between random fluctuations 
and active swimming. Drives them into out-of-equilibirum status. 

**Motivation**: potential to pick up and deliver nanoscopic objects. 
Bioremediation, drug delivery, gene therapy. 

**Aim of paper**: step-by-step description of modelling active Brownian particle 
in homogeneous and complex environments. Use two-dimensional stochastic 
differential equations. Solve equations numerically with simple finite-
difference algorithms. Then simulate motion when several obstacles are present,
use reflective boundary conditions. (They provide MATLAB code but we will 
implement this in Python.)

## Mathematical Model 
We will model a two-dimensional homogeneous environment (note to self: perhaps 
discuss three-dimensional model with Dr. Karani?). Model is three combined 
processes: 
1. Random diffusion process
2. Self propelling force 
3. With chiral active particles, a torque

The position $[x(t), y(t)]$ of a spherical microscopic particle with radius 
$R$ (we will model a rectangle instead, but this is not relevant now) undergoes 
Brownian diffusion (TODO: look up where this constant shows up later. Looks like 
in the denominator we have some sort of area term, maybe could just swap 
out by some sort height times width?)
$$D_T = \frac{k_BT}{6\pi\eta R^3}$$

$k_B$ is the Boltzmann constant, $T$ is the temperature, $\eta$ is the fluid 
viscosity. Particle self-propulsion yields directed component of the motion 
with speed $v$ we will assume to be constant. Direction depends on particle 
orientation $\phi(t)$. Rotational diffusion occurs with rotational diffusion 
coefficient (TODO look this up as well):
$$D_R = \frac{k_B T}{8\pi \eta R^3}$$

For a chiral active particle $\phi(t)$ (the particle orientation itself) rotates 
with angular velocity $\Omega$ as a consequence of the torque. With $v$ this 
reorientation leads to rotation around effective external axis (particle moves 
in circular motion). Sign of $\Omega$ determines chirality (TODO look this up).

So the **functions** are:
- $x(t),y(t)$: Cartesian coordinates of 2-D position 
- $\phi(t)$ orientation of molecule

This yields a set of Langevin equations (TODO Look this up) is 
$$
\frac{d}{dt}\phi(t) = \Omega + \sqrt{2D_R}
$$

$$
\frac{d}{dt}x(t) = v\cos \phi(t) + \sqrt{2D_T }W_x
$$

$$
\frac{d}{dt}x(t) = v\cos \phi(t) + \sqrt{2D_T }W_x
$$

$W_\phi$, $W_x$, $W_y$ are white noise processes.
Inertial processes are ignored (TODO how to enable
inertia? is this just because it's a single derivative?) because of low Reynolds
number. 

This is what the papers give us. We will have to incorporate a term for 
particle interactions. Probably some sort of term will be added like this 
(not exactly correct, see section on reflective boundaries)

$$
\frac{d}{dt}x(t) = v\cos \phi(t) + \sqrt{2D_T }W_x - \dot{x}(t_{i-1})U(x, y)
$$

where $U$ is one of $0,1$ depending on whether the particle is overlapping with 
another; i.e., whether particle is colliding (assuming perfectly elastic 
collisions, I suppose. In this case we might have to take into account 
some momentum exchange if we want to give the particles custom mass).

## Finite Difference Equations 
Approximate a solution to the above differential equation using a discrete 
time sequence 
$$[\phi_i, x_i, t_i]\approx [\phi(t_i), x(t_i),y(t_i)]$$

where this is the solution of the corresponding set of difference equations 
evaluated at discrete, regular time steps $t_i = i\Delta t$, with $\Delta t$
a sufficiently small time step. White noise terms must be dealt with carefully. 

In the differential equations we substitute 
- $\phi(t) \rightarrow \phi_i$
- $x(t) \rightarrow x_i$
- $y(t) \rightarrow y_i$
- $\frac{d}{dt}\phi(t) \rightarrow (\phi_i - \phi_{i-1})/\Delta t$
- $\frac{d}{dt}x(t) \rightarrow (x_i - x_{i-1})/\Delta t$
- $\frac{d}{dt}y(t) \rightarrow (y_i - y_{i-1})/\Delta t$
- $W_\phi \rightarrow w_{\phi,i}/\sqrt{\Delta t}$
- $W_x \rightarrow w_{x,i}/\sqrt{\Delta t}$
- $W_y \rightarrow w_{y,i}/\sqrt{\Delta t}$

$w_{\phi,i}$, $w_{x,i}$, $w_{y,i}$, are _uncorrelated_ sequences of random 
numebrs taken from a Gaussian distribution with mean zero and standard deviation 
one. 

Numerical solution is obtained by solving the resulting finite difference 
equation recursively for $[\phi_i,x_i, y_i]$, using values $[\phi_{i-1}, 
x_{i-1}, y_{i-1}]$ (TODO work out how the following result so you can do it 
yourself for the case with self-interaction)

$$
\phi_i = \phi_{i-1} + \Omega\Delta t + \sqrt{2D_R \Delta t}w_{\phi,i}
$$

$$
x_i = x_{i-1} + v\cos \phi_{i-1}\Delta t + \sqrt{2D_T \Delta t}w_{x,i}
$$

$$
y_i = y_{i-1} + v\cos \phi_{i-1}\Delta t + \sqrt{2D_T \Delta t}w_{x,i}
$$

(This is a first-order integration method; can use higher order algorithms to 
obtain faster solution convergence. TODO what does this mean lol)


## Homogeneous Environment
$v=0$ defaults to pure diffusive Brownian particle. The mean squared 
displacement (MSD) quantifies how a particle moves from its initial position. 
A discrete calculation from a trajectory $[x_n,y_n]$ sampled at discrete times 
$t_n$ with a stime step $\Delta t$ as 
$$
\text{MSD}(m\Delta t) = \langle  \rangle
$$
where the angle brackets denote the mean (supposedly as $m\Delta t$ varies).
(TODO internalize translation from tau to $m\Delta t$ in paper).

## Complex Environments 
### Reflective Boundaries
Whenver an active particle contacts an obstacle, it slides along the obstacle 
until its orientation points away from it. Numerically we can model this 
process using reflective boundaries. 

We will implement this by updating at each time stop the 
particle position from $\mathbf{r}_{i-1} = [x_{i-1}, y_{i-1}]$ to 
$\mathbf{r}_{i} = [x_i,y_u]$ by the following algorithm 
1. Tentatively update the particle to $\bar{\mathbf{r}}$ as given by the above 
   algorithm. 
2. If $\bar{\mathbf{r}}$ is not in any obstacle, set $\hat{\mathbf{r}}_i = 
   \mathbf{r}_i $ and move on to the next time step. 
3. Otherwise (i.e., if $\hat{\mathbf{r}}_i$) is inside some obsstacle, 
   1. Calculate the intersection point $\mathbf{p} = [x_p, y_p]$ between the 
      boundary and the line from $\mathbf{r}_{i-1}$ to $\mathbf{r}_i$. 
   2. Calculate the straight line $l$ tangent to the obstacle at $\mathbf{p}$ 
      with tangent unit vector $\hat{\mathbf{t}}$ and normal unit vector 
      $\hat{\mathbf{n}}$ outgoing from the obstacle. 
   3. Calculate $\mathbf{r}_i$ by reflecting $\bar{\mathbf{r}}_i$ on $l$ 
      so that 
      $$
      \mathbf{r}_i = \bar{\mathbf{r}}_i - 2[(\hat{\mathbf{r}_i})\cdot \bar{
      \mathbf{n}}] \bar{\mathbf{n}}
      $$
      where $(\bar{\mathbf{r}}_i - \mathbf{p}) \cdot \hat{\mathbf{n}}$.

For this method to work, the average spatial increment of a simulated 
trajectory is small compared to the characteristic length scale of the 
obstacles (TODO what does this mean?). This way we can consider one boundary
at a time and also approximate the boundary with its tangent straight line. 
If the time step $\Delta t$ is too large, this approach can lead to numerical 
instability around sharp corners in the boundaries, where multiple reflections 
may take place, or an obstacle wall that is too thin, where the particle's 
trajectory could unnaturally pass through the obstacle. 



# Implementation Notes 
Use scipy (?) `odeint` perhaps instead of algorithm that they provided? Check 
with Dr. Karani if so; probably more instructive to devise own algorithm. 

- Swimmer object: 
  - instance variables can be mass (but mass defined in terms 
    inertia and we are dealing with low Reynolds number, perhaps this is silly.
    Or maybe cool? TODO discuss with Dr. Karani a custom reynolds number), 
    current coordinates, velocity (length of vector $[dx/dt, dy/dt]$),
    angular velocity, dimensions of rectangle?
  - method can be `.step()` or something to solve for next timestep.
  - going to have to be able to use global handler (somehow minus itself) 
    to compute $U(x,y)$
- Boundary object: circular, but allow any curve, perhaps parametrically 
  defined.
- Global handler for all the swimmers as well as boundary. 
  Data type: dictionary (recall can 
  call `.keys` to obtain keys to iterate over)? set?
- Creating animation: TODO. This will be frustrating, no doubt. Refer to your 
  matplotlib frame-by-frame video of heat equation solution from a while 
  ago.
- Somewhere figure out where in the API you're gonna put formulae like the MSD.



