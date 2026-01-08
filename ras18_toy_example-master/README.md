# Force-based variable impedance learning

This code shows a 2D spring-damper system (MSD) simulation subject to an 
external forces to simulate a framework that integrates force sensing 
systems and variable impedance control to learn force-based variable 
stiffness skills. Such skills rely on an estimation of the stiffness 
that is computed from the human demonstrations, which is then used along 
with the sensed forces to encode a probabilistic model of the task.



### REFERENCE
Please acknowledge the authors in any academic publications that used parts of these codes.

#### [1] [Force-based variable impedance learning for robotic manipulation](https://www.sciencedirect.com/science/article/pii/S0921889018300125)
```
@article{abudakka2018force,
    title={Force-based variable impedance learning for robotic 
          manipulation},
    author={Abu-Dakka, Fares J and Rozo, Leonel and Caldwell, Darwin G},
    journal={Robotics and Autonomous Systems},
    volume={109},
    pages={156--167},
    year={2018}
}
```
## Installation instructions

* ``` git clone git@gitlab.com:variable_impedance/ras18_toy_example.git ```

```
Copyright (c) 2017-18 Fares J. Abu-Dakka

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.
 
This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.
 
You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
