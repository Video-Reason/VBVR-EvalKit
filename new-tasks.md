# Proposed New Tasks for VMEvalKit

This document presents **80+ new cognitive reasoning task ideas** for evaluating video generation models. These tasks extend VMEvalKit's evaluation capabilities across diverse reasoning domains beyond the existing benchmarks (Chess, Maze, Raven's Matrices, 3D Rotation, Sudoku, Clock, Object Subtraction, etc.).

## Table of Contents

1. [Overview](#overview)
2. [Physics & Physical Reasoning](#physics--physical-reasoning)
3. [Mathematical & Logical Reasoning](#mathematical--logical-reasoning)
4. [Causal & Temporal Reasoning](#causal--temporal-reasoning)
5. [Game & Strategy Reasoning](#game--strategy-reasoning)
6. [Spatial & Geometric Reasoning](#spatial--geometric-reasoning)
7. [Real-World Practical Reasoning](#real-world-practical-reasoning)
8. [Biological & Natural Reasoning](#biological--natural-reasoning)
9. [Unconventional & Creative Reasoning](#unconventional--creative-reasoning)
10. [Priority Recommendations](#priority-recommendations)
11. [Research Context](#research-context)
12. [Implementation Considerations](#implementation-considerations)

---

## Overview

### Design Principles

Each proposed task follows the VMEvalKit paradigm:
- **Initial Image**: Shows the problem state
- **Text Prompt**: Describes the goal/transformation
- **Video Output**: Demonstrates the solution process
- **Final Frame**: Matches expected ground truth for automated scoring

### Evaluation Criteria for Task Selection

| Criterion | Description |
|-----------|-------------|
| **Deterministic Solution** | Task has exactly one correct answer (or small set of valid answers) |
| **Visual Clarity** | Initial and final states are unambiguous |
| **Procedural Generation** | Can generate unlimited unique instances programmatically |
| **Scoring Feasibility** | Final frame comparison or MLLM evaluation is reliable |
| **Reasoning Depth** | Tests genuine cognitive capability, not just visual generation |

---

## Physics & Physical Reasoning

Tasks testing intuitive physics, mechanical reasoning, and physical causality.

### 1. Balance Point Prediction

**Description**: Given an irregularly shaped object (or composite object made of different materials), predict where to place a fulcrum so the object balances horizontally.

**Initial Image**: A 2D side-view of an asymmetric object (e.g., an L-shaped beam, a hammer, or a composite bar with varying densities indicated by color) resting on a surface, with a triangular fulcrum nearby.

**Video Demonstrates**: The object being placed on the fulcrum at the correct balance point, achieving stable horizontal equilibrium.

**Reasoning Tested**: Center of mass intuition, density/weight distribution understanding, torque balancing.

**Difficulty Scaling**:
- Easy: Uniform density, simple shapes
- Medium: Composite objects with 2-3 density regions
- Hard: Complex shapes with non-obvious center of mass

---

### 2. Domino Chain Completion

**Description**: Given a partially arranged domino chain with a gap, determine the correct placement of missing dominoes to complete a successful chain reaction.

**Initial Image**: A bird's-eye or angled view of dominoes arranged in a pattern with 1-3 missing pieces (gaps shown with dotted outlines or marked positions).

**Video Demonstrates**: The missing dominoes being placed, then the first domino being pushed, triggering a complete chain reaction through all pieces.

**Reasoning Tested**: Collision propagation, spacing/angle requirements, sequential cause-and-effect understanding.

**Difficulty Scaling**:
- Easy: Linear chain, single missing domino
- Medium: Curved path, 2-3 missing dominoes
- Hard: Branching paths, critical placement angles

---

### 3. Pulley System Load Lifting

**Description**: Given a pulley configuration and a heavy load, determine which rope to pull (and in which direction) to lift the load efficiently.

**Initial Image**: A mechanical pulley system (single, double, or compound) with ropes, a weight attached, and an arrow or hand indicating potential pull points.

**Video Demonstrates**: The correct rope being pulled in the proper direction, causing the load to rise smoothly.

**Reasoning Tested**: Mechanical advantage understanding, force direction in pulley systems, rope tension propagation.

---

### 4. Projectile Landing Prediction

**Description**: Given a cannon/launcher at a specific angle and power setting, predict where a projectile will land on a terrain with obstacles.

**Initial Image**: A side-view scene with a cannon at a set angle, power meter indicated, and a landscape with platforms, gaps, or targets at various distances.

**Video Demonstrates**: The projectile being launched, following a parabolic arc, and landing at the predicted/correct location.

**Reasoning Tested**: Trajectory prediction, gravity effects, initial velocity decomposition, obstacle clearance.

---

### 5. Gear Train Direction

**Description**: Given a system of interconnected gears of various sizes, determine which direction the final gear will rotate when the first gear is turned.

**Initial Image**: A top-down or angled view of 3-6 meshed gears of different sizes, with an arrow showing the input rotation direction on the first gear, and a question mark on the final gear.

**Video Demonstrates**: The first gear rotating, causing sequential rotation through the chain, with the final gear spinning in the correct direction.

**Reasoning Tested**: Rotational direction reversal with meshing gears, gear ratio intuition, mechanical linkage understanding.

**Difficulty Scaling**:
- Easy: 3 gears in a line
- Medium: 5 gears with different sizes
- Hard: 7+ gears with idler gears and branching

---

### 6. Fluid Fill Level (Communicating Vessels)

**Description**: Given connected vessels of different shapes, predict the final water level when liquid is poured into one container.

**Initial Image**: Two or more connected containers of varying widths and heights, with an indicated amount of water about to be poured into one.

**Video Demonstrates**: Water being poured in and distributing through the system, settling at the same height across all containers regardless of shape.

**Reasoning Tested**: Pascal's principle, hydrostatic pressure, conservation of volume, counter-intuitive equilibrium.

---

### 7. Ramp and Collision Ordering

**Description**: Given multiple balls of different masses/sizes released simultaneously from ramps of varying heights and angles, predict the order they reach the bottom or collide.

**Initial Image**: Parallel or converging ramps with differently colored/sized balls positioned at starting points, with a finish line or collision zone marked.

**Video Demonstrates**: Balls being released and rolling down, reaching the destination in the predicted sequence.

**Reasoning Tested**: Acceleration due to gravity, friction considerations, mass-independence of gravitational acceleration, energy conversion.

---

### 8. Structural Stability Assessment

**Description**: Given a stack or structure of blocks/shapes, predict whether it will stand or topple, and if it topples, which direction it will fall.

**Initial Image**: A precariously balanced tower or structure of various shaped blocks, some overhanging.

**Video Demonstrates**: Time passing with the structure either remaining stable OR toppling in the predicted direction.

**Reasoning Tested**: Center of gravity, base of support, torque from overhanging masses, stability conditions.

---

### 9. Pendulum Collision Sequence (Newton's Cradle Variant)

**Description**: Given a modified Newton's cradle setup with balls of different masses or spacing, predict the outcome when one or more balls are released.

**Initial Image**: A Newton's cradle variant with labeled masses or irregular spacing, showing which ball(s) will be released.

**Video Demonstrates**: The ball(s) being released, colliding, and the resulting motion pattern.

**Reasoning Tested**: Conservation of momentum and energy, elastic collision outcomes, mass ratio effects.

---

### 10. Lever and Counterweight

**Description**: Given a lever with a known load on one side, determine the correct counterweight or position needed to achieve balance.

**Initial Image**: A see-saw or lever with a weight on one end, with options for counterweight placement.

**Video Demonstrates**: The correct counterweight being placed, causing the lever to balance.

**Reasoning Tested**: Torque calculation (force x distance), lever arm principles, mechanical equilibrium.

---

### 11. Inclined Plane Friction Threshold

**Description**: Given objects of different materials on an adjustable inclined plane, predict at what angle each object will begin to slide.

**Initial Image**: An inclined plane with multiple objects (rubber block, wooden block, ice cube) and an angle indicator.

**Video Demonstrates**: The plane tilting gradually until each object slides at its predicted threshold angle.

**Reasoning Tested**: Static vs. kinetic friction, coefficient of friction intuition, normal force dependency on angle.

---

### 12. Buoyancy and Floating Depth

**Description**: Given objects of different densities placed in a liquid, predict their floating depth or whether they sink entirely.

**Initial Image**: A tank of liquid and several objects of known or implied densities about to be dropped in.

**Video Demonstrates**: Objects being dropped and settling at their correct floating depths or sinking.

**Reasoning Tested**: Archimedes' principle, density comparison, displaced volume calculation.

---

### 13. Elastic vs. Inelastic Collision

**Description**: Given two objects about to collide with known properties (elastic ball vs. clay ball), predict their post-collision velocities and directions.

**Initial Image**: Two objects approaching each other with velocity vectors indicated and labels showing their material properties.

**Video Demonstrates**: The collision occurring with appropriate outcome (bouncing apart vs. sticking together).

**Reasoning Tested**: Conservation of momentum, energy dissipation in inelastic collisions, coefficient of restitution.

---

### 14. Rube Goldberg Missing Link

**Description**: Given a complex chain-reaction machine with one missing component, identify what element completes the sequence.

**Initial Image**: A partial Rube Goldberg machine with a visible gap in the chain.

**Video Demonstrates**: The missing component being inserted, then the machine running from start to finish.

**Reasoning Tested**: Multi-step causal reasoning, energy transfer between mechanisms, spatial and temporal sequencing.

---

### 15. Pressure and Container Deformation

**Description**: Given a sealed flexible container with gas and an external pressure change, predict how the container deforms.

**Initial Image**: A balloon or flexible container in one environment, with an indicator showing it will move to a different pressure environment.

**Video Demonstrates**: The container expanding or compressing appropriately as it transitions.

**Reasoning Tested**: Boyle's Law intuition, gas pressure and volume relationship.

---

## Mathematical & Logical Reasoning

Tasks testing arithmetic, geometry, logic, and symbolic manipulation.

### 1. Visual Arithmetic

**Description**: Display groups of countable objects with an arithmetic operator. The model must animate counting and show the computed result.

**Initial Image**: Two groups of objects (e.g., 3 apples + 2 apples) with an operator symbol and empty answer box.

**Video Demonstrates**: Objects being counted or combined visually, with the numerical answer appearing.

**Reasoning Tested**: Counting/enumeration, basic arithmetic operations, visual-to-numerical mapping.

**Difficulty Scaling**:
- Easy: 1-5 objects, addition only
- Medium: 5-10 objects, addition and subtraction
- Hard: 10-20 objects, mixed operations

---

### 2. Venn Diagram Set Operations

**Description**: Display overlapping circles with elements, and identify/highlight the result of a set operation.

**Initial Image**: Two overlapping circles (Venn diagram) with labeled elements and a question asking for intersection, union, or difference.

**Video Demonstrates**: Highlighting of relevant regions with elements in the answer region being identified.

**Reasoning Tested**: Set theory understanding, spatial region identification, logical classification.

---

### 3. Number Sequence Completion

**Description**: Display a sequence of numbers with a missing element. Identify the pattern and fill in the missing number.

**Initial Image**: A row of numbers with one position marked as "?" (e.g., [2, 4, 6, ?, 10, 12]).

**Video Demonstrates**: Visual cues showing pattern recognition, with the missing number appearing.

**Reasoning Tested**: Pattern recognition, arithmetic/geometric progressions, mathematical induction.

**Difficulty Scaling**:
- Easy: Simple arithmetic progressions
- Medium: Geometric progressions, alternating patterns
- Hard: Fibonacci, quadratic sequences

---

### 4. Graph Shortest Path

**Description**: Display a weighted graph with source and destination nodes. Show the shortest path between them.

**Initial Image**: Nodes connected by edges with weights, source node in green, destination in red.

**Video Demonstrates**: Path exploration with edges lighting up, final shortest path highlighted with total cost.

**Reasoning Tested**: Graph traversal, optimization reasoning, path cost calculation.

---

### 5. Fraction Visualization and Comparison

**Description**: Display two fractions using pie charts or bar representations. Compare them or compute their sum.

**Initial Image**: Two fractions shown as divided shapes with numerical representation below.

**Video Demonstrates**: Visual alignment/comparison of shaded regions with result appearing.

**Reasoning Tested**: Fraction understanding, visual comparison, common denominator reasoning.

---

### 6. Balance Scale Equation

**Description**: Display a balance scale with known and unknown weights. Determine the value of the unknown weight.

**Initial Image**: A balance scale with known weights and an unknown weight labeled "x" or "?".

**Video Demonstrates**: The scale adjusting, unknown weight value being revealed, scale reaching equilibrium.

**Reasoning Tested**: Algebraic reasoning, equation solving, balance/equality concepts.

**Difficulty Scaling**:
- Easy: x + 3 = 7 style
- Medium: 2x + 3 = 11 style
- Hard: Multiple unknowns

---

### 7. Permutation/Arrangement Counting

**Description**: Display a set of objects with constraints and show how many unique arrangements are possible.

**Initial Image**: Distinct objects, arrangement slots, constraints (e.g., "Red must be first").

**Video Demonstrates**: Systematic enumeration or tree diagram, counter incrementing, final count.

**Reasoning Tested**: Combinatorial reasoning, constraint satisfaction, systematic enumeration.

---

### 8. Geometric Transformation Proof

**Description**: Display a geometric figure with marked properties. Show the steps to prove a relationship.

**Initial Image**: A geometric figure with given information marked (equal sides, angles, parallel lines).

**Video Demonstrates**: Step-by-step highlighting of relevant properties, intermediate conclusions, final proof.

**Reasoning Tested**: Deductive reasoning, geometric theorem application, multi-step logical chains.

---

### 9. Number Base Conversion

**Description**: Display a number in one base and convert it to another base, showing the calculation.

**Initial Image**: A number in source base (e.g., "1101" base 2) with target base indicator.

**Video Demonstrates**: Place value breakdown, multiplication and addition steps, final answer.

**Reasoning Tested**: Positional notation understanding, base/radix comprehension, systematic calculation.

---

### 10. Logic Gate Circuit Evaluation

**Description**: Display a logic circuit with gates (AND, OR, NOT, XOR) and input values. Trace through and show the output.

**Initial Image**: Circuit diagram with logic gates, input values (0 or 1), and empty output indicator.

**Video Demonstrates**: Signal propagation through the circuit, each gate's output computed in sequence, final output.

**Reasoning Tested**: Boolean logic, sequential evaluation, truth table application.

**Difficulty Scaling**:
- Easy: 2 inputs, 1-2 gates
- Medium: 3-4 inputs, 3-4 gates
- Hard: 4+ inputs, 5+ gates with parallel paths

---

## Causal & Temporal Reasoning

Tasks testing cause-and-effect, temporal sequencing, and state transitions.

### 1. Domino Chain Reaction

**Description**: Given dominoes with specific spacing and angles, predict which will fall and which will remain standing.

**Initial Image**: Top-down view of dominoes with one marked as "about to fall", some gaps too wide to propagate.

**Video Demonstrates**: Causal propagation of falling dominoes, correct identification of chain breaks.

**Reasoning Tested**: Causal chain reasoning, physical intuition about momentum and spacing.

---

### 2. Traffic Light State Machine

**Description**: Given a multi-way intersection with traffic lights and cars, predict the sequence of light changes and car movements.

**Initial Image**: Top-down intersection view with traffic lights showing current states and cars waiting.

**Video Demonstrates**: Correct state machine transitions, cars moving only on green, proper sequencing.

**Reasoning Tested**: Finite state machine understanding, temporal sequencing, rule-based reasoning.

---

### 3. Cooking Process Completion

**Description**: Given ingredients in various preparation states, determine the correct sequence of cooking steps.

**Initial Image**: Kitchen scene with ingredients at different stages, target dish silhouette, cooking tools.

**Video Demonstrates**: Correct ordering of preparation steps, respecting irreversibility constraints.

**Reasoning Tested**: Temporal ordering with irreversibility, causal dependencies, process planning.

---

### 4. Plant Growth Timelapse

**Description**: Given environmental conditions (sunlight direction, water), predict how a plant will grow.

**Initial Image**: Small seedling with sun position and water source indicated.

**Video Demonstrates**: Phototropism (growing toward light), root growth toward water, final plant shape.

**Reasoning Tested**: Biological causal reasoning, temporal process prediction, environmental influence.

---

### 5. Ice Melting Prediction

**Description**: Given ice cubes of different sizes near heat sources, predict the melting sequence and final water levels.

**Initial Image**: Containers with ice cubes of varying sizes, heat source indicators.

**Video Demonstrates**: Smaller ice near heat melts first, correct melting order, final water distribution.

**Reasoning Tested**: Physical causal reasoning, rate prediction, conservation of mass.

---

### 6. Gear Train Rotation

**Description**: Given interconnected gears of different sizes, predict rotation direction and relative speed of each.

**Initial Image**: 2D arrangement of meshing gears with input gear rotation direction marked.

**Video Demonstrates**: Alternating rotation directions, speed ratios based on gear sizes.

**Reasoning Tested**: Mechanical causal chains, ratio reasoning, directional reversal propagation.

---

### 7. Water Flow Network

**Description**: Given pipes with valves, reservoirs, and drains, predict water distribution when a valve opens.

**Initial Image**: Network of pipes with open/closed valves, water source, empty containers.

**Video Demonstrates**: Water flowing through open paths, filling order based on path length and gravity.

**Reasoning Tested**: Path-finding with constraints, fluid dynamics intuition, state change propagation.

---

### 8. Electrical Circuit State

**Description**: Given a circuit with switches, bulbs, and batteries, predict which bulbs light when switches toggle.

**Initial Image**: Circuit diagram with batteries, switches, and light bulbs in current states.

**Video Demonstrates**: Correct causal chain from switch to circuit to bulb, series vs parallel behavior.

**Reasoning Tested**: Logical circuit reasoning, state machine transitions, electrical flow understanding.

---

### 9. Falling Sand/Particle Simulation

**Description**: Given particles suspended on platforms, predict how they settle after gravity is applied.

**Initial Image**: Grid with particles (sand, water) on platforms with empty spaces below.

**Video Demonstrates**: Particles falling, sand forming piles, water spreading horizontally.

**Reasoning Tested**: Physics simulation prediction, emergent behavior from simple rules.

---

### 10. Infection Spread Prediction

**Description**: Given a network with some infected nodes and barriers, predict the spread pattern over time steps.

**Initial Image**: Grid of nodes (infected, healthy, immune) with connection lines and barriers.

**Video Demonstrates**: Spreading through valid connections, immunity blocking transmission.

**Reasoning Tested**: Graph-based causal propagation, rule application over time steps.

---

## Game & Strategy Reasoning

Tasks testing strategic thinking in game contexts.

### 1. Connect-4 Winning Move

**Description**: Given a Connect-4 board where one player can win in one move, drop the winning piece.

**Initial Image**: 7x6 Connect-4 grid with red and yellow pieces in a near-winning configuration.

**Video Demonstrates**: Piece dropping into correct column, falling with gravity, completing 4-in-a-row.

**Reasoning Tested**: Pattern recognition, spatial reasoning across multiple axes, strategic foresight.

---

### 2. Tower of Hanoi

**Description**: Given an initial Tower of Hanoi configuration with 2-3 disks, show the sequence to transfer all disks to target peg.

**Initial Image**: Three pegs with graduated disks stacked on one peg, target peg marked.

**Video Demonstrates**: Sequential disk movements following rules, final configuration on target peg.

**Reasoning Tested**: Recursive/hierarchical planning, constraint satisfaction, multi-step reasoning.

**Difficulty Scaling**:
- Easy: 2 disks (3 moves)
- Medium: 3 disks (7 moves)
- Hard: 4 disks (15 moves)

---

### 3. Minesweeper Safe Move

**Description**: Given a partially revealed Minesweeper grid, reveal a guaranteed-safe cell based on number clues.

**Initial Image**: Minesweeper grid with numbers indicating adjacent mine counts, some unrevealed cells.

**Video Demonstrates**: Cursor moving to safe cell, cell being revealed.

**Reasoning Tested**: Logical deduction from constraints, probabilistic reasoning, spatial analysis.

---

### 4. 2048 Single Swipe

**Description**: Given a 2048 game board, perform the optimal swipe to maximize score gain.

**Initial Image**: 4x4 grid with numbered tiles (powers of 2) in various positions.

**Video Demonstrates**: All tiles sliding simultaneously, same-value tiles merging.

**Reasoning Tested**: Multi-object trajectory prediction, value aggregation, optimization.

---

### 5. N-Queens Single Placement

**Description**: Given an N-Queens board with N-1 queens placed correctly, place the final queen in the only valid position.

**Initial Image**: Chess-like board with queens already placed, attack lines visualized.

**Video Demonstrates**: Analysis of threatened squares, queen moving to safe square.

**Reasoning Tested**: Constraint propagation, diagonal/linear threat recognition, elimination reasoning.

---

### 6. Othello/Reversi Capture Move

**Description**: Given an Othello board, place a piece that maximizes captures.

**Initial Image**: 8x8 Othello board with black and white pieces, valid positions marked.

**Video Demonstrates**: Piece placed on optimal square, captured pieces flipping sequentially.

**Reasoning Tested**: Multi-directional line scanning, counting and optimization, enclosure mechanics.

---

### 7. Tetris Line Clear

**Description**: Given a Tetris playfield with a falling piece, land it to clear complete lines.

**Initial Image**: Tetris playfield with accumulated blocks, current falling tetromino, near-complete rows.

**Video Demonstrates**: Piece rotating to optimal orientation, moving to correct column, lines clearing.

**Reasoning Tested**: Shape fitting, pattern completion, physics understanding.

---

### 8. Checkers Forced Jump

**Description**: Given a checkers board where a capture is available, demonstrate the jumping move(s).

**Initial Image**: 8x8 checkerboard with red and black pieces, jump opportunity present.

**Video Demonstrates**: Piece jumping diagonally over opponent, captured piece removed, chain-jumps if available.

**Reasoning Tested**: Diagonal movement rules, forced-move understanding, chain reasoning.

---

### 9. Resource Allocation

**Description**: Given containers with capacities and resources with sizes, show optimal distribution.

**Initial Image**: Multiple containers with labeled capacities, resources with labeled sizes.

**Video Demonstrates**: Resources being placed into containers respecting constraints.

**Reasoning Tested**: Bin-packing, constraint satisfaction, proportional reasoning.

---

### 10. Go Capture (Atari Resolution)

**Description**: Given a Go board with a group in "atari" (one liberty), demonstrate the capturing or escape move.

**Initial Image**: Go board with a group clearly in atari, single liberty point visible.

**Video Demonstrates**: Stone placed on decisive point, surrounded stones removed (if capture).

**Reasoning Tested**: Liberty counting, connected component reasoning, life-and-death analysis.

---

## Spatial & Geometric Reasoning

Tasks testing mental transformation, spatial visualization, and geometric understanding.

### 1. Origami Fold Prediction

**Description**: Given a flat square with marked fold lines and sequence indicators, demonstrate the folding process.

**Initial Image**: Flat paper with dashed (valley) and dot-dash (mountain) fold lines, numbered arrows.

**Video Demonstrates**: Sequential folding, layers accumulating, final 3D form.

**Reasoning Tested**: 2D-to-3D spatial transformation, sequential operation understanding, fold semantics.

---

### 2. Paper Punch Hole Pattern

**Description**: Paper is folded multiple times and punched. Show the unfolded paper with all hole positions.

**Initial Image**: Folded paper (side view showing layers), hole punch position marked.

**Video Demonstrates**: Unfolding process in reverse, holes appearing symmetrically, final flat paper with all holes.

**Reasoning Tested**: Symmetry reasoning across fold lines, multiplicative spatial thinking, layer tracking.

---

### 3. Orthographic Projection Matching

**Description**: Given a 3D object, generate the correct front, side, and top 2D views.

**Initial Image**: 3D object with one or two orthographic views provided, empty slot for missing view.

**Video Demonstrates**: Camera rotating around object, pausing at orthographic positions, complete three-view set.

**Reasoning Tested**: 3D-to-2D projection, multi-view correspondence, hidden line reasoning.

---

### 4. Tangram Assembly

**Description**: Given tangram pieces and a target silhouette, assemble pieces to match the target.

**Initial Image**: Seven tangram pieces scattered, target silhouette outline shown.

**Video Demonstrates**: Pieces moving, rotating, flipping into position within silhouette.

**Reasoning Tested**: Spatial decomposition/composition, rotation and reflection, part-whole relationships.

---

### 5. Cross-Section Visualization

**Description**: Given a 3D solid and a cutting plane, show the resulting 2D cross-section shape.

**Initial Image**: 3D solid (sphere, cone, cylinder) with semi-transparent cutting plane.

**Video Demonstrates**: Cutting plane moving through object, cross-section extracted and displayed flat.

**Reasoning Tested**: 3D slicing visualization, conic sections understanding, spatial interpolation.

---

### 6. Net-to-Solid Folding

**Description**: Given a 2D net of a polyhedron, demonstrate folding it into the 3D solid.

**Initial Image**: 2D net layout with faces labeled, fold edges indicated.

**Video Demonstrates**: Faces folding up sequentially, edges aligning, complete 3D polyhedron.

**Reasoning Tested**: 2D-to-3D transformation, edge correspondence, face adjacency understanding.

---

### 7. Reflection and Symmetry Completion

**Description**: Given a partial pattern and symmetry axis, complete the pattern.

**Initial Image**: Partial 2D pattern with clearly marked symmetry axis, grid overlay.

**Video Demonstrates**: Elements being mirrored across axis, pattern growing to completion.

**Reasoning Tested**: Reflection transformation, rotational symmetry, pattern extrapolation.

---

### 8. 3D Puzzle Assembly (Soma Cube)

**Description**: Given 3D polycube pieces, assemble them into a target shape.

**Initial Image**: Multiple 3D polycube pieces displayed separately, target shape as wireframe.

**Video Demonstrates**: Pieces moving and rotating in 3D, fitting together progressively.

**Reasoning Tested**: 3D spatial manipulation, volumetric reasoning, combinatorial problem-solving.

---

### 9. Shadow Projection Reasoning

**Description**: Given a 3D object and light source direction, generate the correct shadow shape.

**Initial Image**: 3D object with arrow indicating light direction, flat ground plane.

**Video Demonstrates**: Light rays emanating, shadow shape forming on ground plane.

**Reasoning Tested**: Projection geometry, 3D-to-2D mapping, occluded region reasoning.

---

### 10. Isometric Block Counting

**Description**: Given an isometric view of stacked blocks with some hidden, determine the total count.

**Initial Image**: Isometric view of block structure, "Total cubes: ?" counter.

**Video Demonstrates**: Structure rotating or becoming transparent to reveal hidden blocks, counting.

**Reasoning Tested**: Hidden object inference, 3D reasoning from 2D projection, spatial enumeration.

---

## Real-World Practical Reasoning

Tasks testing common sense, everyday problem-solving, and practical knowledge.

### 1. Tool Selection

**Description**: Given a problem (loose screw, stuck jar lid), select the correct tool from a set.

**Initial Image**: Workspace with multiple tools and a clearly identifiable problem object.

**Video Demonstrates**: Correct tool moving toward problem object and performing appropriate action.

**Reasoning Tested**: Object affordance understanding, problem-tool matching, functional reasoning.

**Difficulty Scaling**:
- Easy: Single obvious tool choice
- Medium: Multiple plausible tools
- Hard: Tool combinations or non-obvious solutions

---

### 2. Container Opening

**Description**: Given a container with a specific opening mechanism, demonstrate the correct sequence.

**Initial Image**: Closed container (screw-top, flip-top, childproof, ziplock).

**Video Demonstrates**: Container opening through correct physical motion sequence.

**Reasoning Tested**: Physical mechanism understanding, sequential action planning, everyday object knowledge.

---

### 3. Hazard Identification

**Description**: Given a scene with safety hazards, identify and correct them.

**Initial Image**: Room/kitchen/workshop with hazards (water near outlet, knife on edge, tripping hazard).

**Video Demonstrates**: Hazards highlighted/circled, then corrected to safe state.

**Reasoning Tested**: Safety reasoning, risk assessment, causal reasoning about consequences.

---

### 4. Queue/Line Convention

**Description**: Given a queue scene, determine where a new person should stand or who is served next.

**Initial Image**: Queue at bank/store with numbered tickets, priority indicators, multiple lines.

**Video Demonstrates**: Highlighted path showing where new arrival should go, or next person indicated.

**Reasoning Tested**: Social convention understanding, fairness reasoning, signage interpretation.

---

### 5. Assembly Instruction Following

**Description**: Given a partially assembled item and instruction diagram, show the next piece being placed.

**Initial Image**: Partially assembled furniture/toy with instruction diagram showing next step.

**Video Demonstrates**: Next component moving to and attaching at correct position with correct orientation.

**Reasoning Tested**: Instruction interpretation, part-to-whole reasoning, orientation matching.

---

### 6. Recipe Step Completion

**Description**: Given a cooking scene mid-recipe with visible recipe card, show the next cooking action.

**Initial Image**: Kitchen counter with ingredients in various states, visible recipe step.

**Video Demonstrates**: Correct action being performed (cracking eggs, stirring, pouring).

**Reasoning Tested**: Procedural reasoning, state tracking, ingredient-action mapping.

---

### 7. Anomaly Detection ("What's Wrong")

**Description**: Given a scene with something contextually incorrect, identify and correct the error.

**Initial Image**: Everyday scene with anomaly (clock with 13 hours, fork on wrong side, shoes on wrong feet).

**Video Demonstrates**: Anomaly highlighted, then corrected to proper state.

**Reasoning Tested**: Contextual appropriateness, world knowledge, visual anomaly detection.

---

### 8. Traffic Rule Reasoning

**Description**: Given a traffic scenario at an intersection, determine which vehicle(s) proceed next.

**Initial Image**: Intersection with vehicles, traffic signs/signals, pedestrians.

**Video Demonstrates**: Correct vehicle(s) highlighted and moving through intersection.

**Reasoning Tested**: Traffic rule understanding, right-of-way reasoning, multi-agent coordination.

---

### 9. Weight Balance / Stability

**Description**: Given an unstable stacking arrangement, show how to rearrange for stability.

**Initial Image**: Objects stacked precariously (heavy on light, overhanging edges).

**Video Demonstrates**: Objects rearranging to achieve stable configuration.

**Reasoning Tested**: Intuitive physics, center of mass reasoning, weight-to-base relationships.

---

### 10. Electrical/Plumbing Connection

**Description**: Given a partially connected system, show the correct connection to complete it.

**Initial Image**: Incomplete circuit board or pipes that need connecting.

**Video Demonstrates**: Missing connection being made, system becoming functional.

**Reasoning Tested**: Flow/circuit reasoning, connection compatibility, input/output mapping.

---

## Biological & Natural Reasoning

Tasks testing understanding of living systems, natural processes, and scientific reasoning.

### 1. Phototropism Response

**Description**: Given a plant with a directional light source, show how the plant bends toward the light.

**Initial Image**: Young seedling with light source at specific angle.

**Video Demonstrates**: Stem gradually curving toward light, leaves orienting to maximize exposure.

**Reasoning Tested**: Plant growth responses, directional tropisms, biological time-scale transformations.

---

### 2. Predator-Prey Pursuit

**Description**: Given a predator and prey with positions and terrain, predict the chase trajectory.

**Initial Image**: Savanna scene with cheetah and gazelle at marked positions, terrain features.

**Video Demonstrates**: Realistic pursuit paths considering speed, agility, and terrain.

**Reasoning Tested**: Spatial reasoning, animal locomotion, predator-prey dynamics.

---

### 3. Metamorphosis Completion

**Description**: Given an organism at a life stage, generate the transformation to the next stage.

**Initial Image**: Caterpillar on leaf, tadpole in water, or chrysalis on branch.

**Video Demonstrates**: Biological transformation process (tadpole developing legs, losing tail, becoming frog).

**Reasoning Tested**: Developmental biology, sequential morphological changes, life cycle knowledge.

---

### 4. Food Chain Cascade

**Description**: Given an ecosystem with one species removed, show the ripple effects through the food web.

**Initial Image**: Simplified ecosystem illustration with one element marked for removal.

**Video Demonstrates**: Population changes cascading (remove foxes → rabbits explode → grass depletes).

**Reasoning Tested**: Systems thinking, ecological causality, trophic interactions.

---

### 5. Mitotic Cell Division

**Description**: Given a cell at a specific mitosis phase, generate progression through subsequent phases.

**Initial Image**: Cell in prophase, metaphase, or anaphase with visible chromosomes.

**Video Demonstrates**: Correct progression through remaining phases to cytokinesis.

**Reasoning Tested**: Cellular biology, sequential biological processes, subcellular organization.

---

### 6. Seed Dispersal Mechanism

**Description**: Given a mature fruit/seed and environmental conditions, show how the seed disperses.

**Initial Image**: Dandelion head with wind arrows, coconut near water, or burr near animal.

**Video Demonstrates**: Appropriate dispersal mechanism in action.

**Reasoning Tested**: Plant reproduction strategies, physics of dispersal, environment-organism interactions.

---

### 7. Wound Healing Progression

**Description**: Given a tissue injury at a healing stage, generate the biological repair process.

**Initial Image**: Cross-section diagram showing wound with visible tissue layers.

**Video Demonstrates**: Clotting, inflammation, cell migration, tissue regeneration in sequence.

**Reasoning Tested**: Physiological repair processes, temporal biological sequences.

---

### 8. Seasonal Phenology Transition

**Description**: Given a landscape in one season, generate the transition to the next season.

**Initial Image**: Forest scene in late autumn with partially bare trees.

**Video Demonstrates**: Leaves falling, snow accumulating, animals hibernating/migrating.

**Reasoning Tested**: Seasonal cycles, organism responses to environmental cues.

---

### 9. Pollination Pathway

**Description**: Given a flower and pollinator at starting positions, show the pollination process.

**Initial Image**: Bee approaching flower with visible anthers, stigma, and pollen grains.

**Video Demonstrates**: Bee landing, pollen adhering, bee moving to second flower, pollen depositing.

**Reasoning Tested**: Plant reproduction, insect behavior, mutualistic relationships.

---

### 10. Camouflage Adaptation

**Description**: Given an animal in an environment, show how it adapts its appearance for concealment.

**Initial Image**: Chameleon on branch with new background color approaching.

**Video Demonstrates**: Organism's color/texture changing to match environment.

**Reasoning Tested**: Adaptive coloration, organism-environment matching.

---

### 11. Root System Development

**Description**: Given a seed in soil with specific conditions, show root growth patterns.

**Initial Image**: Cross-section with germinated seed, water source, rock obstacles, nutrient zones.

**Video Demonstrates**: Roots growing toward water, around obstacles, branching in nutrient-rich areas.

**Reasoning Tested**: Plant root behavior, multiple tropisms, obstacle navigation.

---

### 12. Skeletal Muscle Contraction

**Description**: Given an anatomical diagram with a muscle highlighted, show the movement when it contracts.

**Initial Image**: Arm diagram with bicep highlighted, limb in extended position.

**Video Demonstrates**: Muscle shortening, bones pivoting, limb moving to flexed position.

**Reasoning Tested**: Musculoskeletal system, lever mechanics, antagonistic muscle pairs.

---

## Unconventional & Creative Reasoning

Tasks testing novel cognitive capabilities beyond standard benchmarks.

### 1. Knot Untying

**Description**: Given a rope with a specific knot, show the sequence of moves to untie it.

**Initial Image**: Rope with recognizable knot (overhand, figure-eight) from clear angle.

**Video Demonstrates**: Systematic loosening and untying through pulls and thread-throughs.

**Reasoning Tested**: 3D spatial reasoning, topological understanding, procedural planning.

---

### 2. Shadow Casting Prediction

**Description**: Given an object and moving light source, show the shadow transitioning.

**Initial Image**: 3D object with point light source at position A casting shadow.

**Video Demonstrates**: Light moving from A to B, shadow morphing accordingly.

**Reasoning Tested**: 3D spatial reasoning, light physics, projection geometry.

---

### 3. Music Box Cylinder Encoding

**Description**: Given a musical staff with a melody, show pins being placed on a music box cylinder.

**Initial Image**: Musical staff with notes, empty cylindrical drum with grid for time and pitch.

**Video Demonstrates**: Pins appearing at correct positions for each note's pitch and timing.

**Reasoning Tested**: Symbol-to-spatial translation, pattern encoding, music notation understanding.

---

### 4. Word Ladder Transformation

**Description**: Given a starting word and target word, show the word transforming through valid intermediates.

**Initial Image**: Starting word (e.g., "COLD") and target word (e.g., "WARM") with empty boxes between.

**Video Demonstrates**: Letters changing one at a time, forming valid words (COLD → CORD → CARD → WARD → WARM).

**Reasoning Tested**: Lexical knowledge, constraint-based search, symbolic manipulation.

---

### 5. Circuit Completion

**Description**: Given an incomplete electrical circuit, show how to connect wires to light specific LEDs.

**Initial Image**: Circuit board with components, loose wires, goal (e.g., "Light up red LED only").

**Video Demonstrates**: Wires connecting step-by-step, target LED lighting up.

**Reasoning Tested**: Logical/electrical reasoning, graph traversal, goal-directed planning.

---

### 6. Origami Unfolding

**Description**: Given a completed origami figure, show it unfolding back to a flat sheet.

**Initial Image**: Fully folded origami figure (crane, boat, frog).

**Video Demonstrates**: Figure systematically unfolding in reverse, revealing crease pattern.

**Reasoning Tested**: 3D-to-2D mental transformation, procedural reasoning in reverse.

---

### 7. Balance Scale Equilibrium

**Description**: Given an unbalanced scale with labeled weights, show which weight to add and where for balance.

**Initial Image**: Tilted balance scale with labeled weights, available weights shown nearby.

**Video Demonstrates**: Correct weight picked up and placed at correct distance, scale leveling out.

**Reasoning Tested**: Physics reasoning (torque), arithmetic, spatial planning.

---

### 8. Isometric Cube Counting

**Description**: Given an isometric view of stacked cubes with some hidden, reveal and count all cubes.

**Initial Image**: Isometric block structure with "Total cubes: ?" counter.

**Video Demonstrates**: Structure rotating or becoming transparent, cubes highlighted and counted.

**Reasoning Tested**: 3D spatial reasoning, hidden object inference, occlusion understanding.

---

### 9. Mirror Symmetry Drawing

**Description**: Given half of a symmetric pattern with a mirror line, draw the missing half.

**Initial Image**: Half of symmetric pattern with mirror line marked.

**Video Demonstrates**: Missing half being drawn stroke-by-stroke.

**Reasoning Tested**: Symmetry understanding, spatial transformation, procedural generation.

---

### 10. Liquid Pouring Physics

**Description**: Given containers with liquids, show pouring to achieve a target mixture/level.

**Initial Image**: Containers with different amounts of colored liquids, target indicated.

**Video Demonstrates**: Pouring sequence to achieve target mixture.

**Reasoning Tested**: Volume reasoning, physical simulation, goal-directed planning.

---

## Priority Recommendations

Based on implementation feasibility, scoring clarity, and novelty, here are the top recommended tasks:

### Tier 1: High Priority (Implement First)

| Rank | Task | Domain | Rationale |
|------|------|--------|-----------|
| 1 | **Gear Train Direction** | Physics | Deterministic, visually clear, tests mechanical reasoning, easy to generate |
| 2 | **Domino Chain Reaction** | Physics/Causal | Clear causality, procedurally generatable, unambiguous final state |
| 3 | **Connect-4 Winning Move** | Games | Single-move, clear win condition, well-understood rules |
| 4 | **Tangram Assembly** | Spatial | Well-defined pieces, exact solution, classic cognitive test |
| 5 | **Logic Gate Circuit** | Math/Logic | Boolean logic, deterministic, easily scales in complexity |

### Tier 2: Medium Priority

| Rank | Task | Domain | Rationale |
|------|------|--------|-----------|
| 6 | **Paper Punch Pattern** | Spatial | Classic IQ test item, symmetry reasoning, clear ground truth |
| 7 | **Tower of Hanoi** | Games | Recursive planning, clear rules, variable difficulty |
| 8 | **Water Flow Network** | Causal | Path reasoning + physics, procedurally generatable |
| 9 | **Balance Scale Equation** | Math | Algebraic reasoning, visual clarity, educational value |
| 10 | **Tool Selection** | Practical | Real-world relevance, affordance reasoning, novel domain |

### Tier 3: Exploratory

| Task | Domain | Notes |
|------|--------|-------|
| Knot Untying | Creative | High difficulty, tests topological reasoning |
| Metamorphosis | Biological | Tests scientific knowledge, longer temporal span |
| Traffic Light State Machine | Causal | Multi-agent, rule-based reasoning |
| Minesweeper Safe Move | Games | Constraint satisfaction, familiar to many |
| Origami Fold Prediction | Spatial | Complex 2D-to-3D transformation |

---

## Research Context

### Current Benchmark Landscape (2024-2025)

Recent work evaluates video generation models on reasoning, not just visual quality:

| Benchmark | Focus | Relevance |
|-----------|-------|-----------|
| **VBench-2.0** | Commonsense, physics realism, motion coherence | Multi-dimensional quality assessment |
| **T2VWorldBench** | World knowledge, physics, causality, culture | 1200 prompts across cognitive domains |
| **WorldSimBench** | Task-driven evaluation via agent performance | Links generation to embodied reasoning |
| **VideoThinkBench** | Video reasoning across multiple domains | ~4.1k reasoning tasks |

### VMEvalKit's Unique Position

VMEvalKit fills a critical gap: **deterministic cognitive task evaluation** where:
- Ground truth can be automatically computed
- Solutions are objectively verifiable
- Tasks require genuine reasoning, not just visual plausibility
- Evaluation doesn't require subjective human judgment

The proposed tasks extend this paradigm across new cognitive domains while maintaining the framework's core strengths.

---

## Implementation Considerations

### Key Files for Adding New Tasks

```
vmevalkit/runner/TASK_CATALOG.py     # Register new task
vmevalkit/tasks/{task_name}/         # Task implementation
  ├── {task_name}_reasoning.py       # Core generation logic
  └── __init__.py                    # Module exports
vmevalkit/eval/gpt4o_eval.py         # Add TASK_GUIDANCE for scoring
docs/ADDING_TASKS.md                 # Implementation checklist
```

### Task Implementation Checklist

1. **Create dataset function**: Returns pairs with `first_image_path`, `final_image_path`, `prompt`, `metadata`
2. **Ensure deterministic solutions**: Each instance has exactly one correct answer
3. **Add to TASK_CATALOG**: Register module path and create function
4. **Add scoring guidance**: Task-specific prompts for GPT-4O evaluation
5. **Test generation**: Verify procedural generation produces valid instances
6. **Validate difficulty scaling**: Ensure easy/medium/hard variants work

### Rendering Approaches

| Approach | Best For | Examples in Codebase |
|----------|----------|---------------------|
| **Matplotlib** | 2D diagrams, grids, graphs | Edit Distance, Shape Sorter |
| **PIL/Pillow** | Photo-realistic scenes, object placement | Object Rearrangement |
| **Chess/Game libraries** | Board games with standard rules | Chess Task |
| **Custom 3D** | Voxel rendering, rotations | Rotation Task |

### Scoring Considerations

- **Final frame comparison**: Works for tasks with unique visual solutions
- **MLLM evaluation**: Needed for tasks with multiple valid approaches
- **Structural comparison**: For tasks where layout matters more than exact pixels
- **Hybrid approaches**: Combine automated checks with MLLM verification

---

## Conclusion

This document presents 80+ task ideas spanning 8 cognitive domains. The priority recommendations balance novelty, feasibility, and alignment with VMEvalKit's evaluation paradigm.

The most promising near-term additions are:
1. **Gear Train Direction** - Mechanical reasoning
2. **Domino Chain** - Physical causality
3. **Connect-4** - Game strategy
4. **Tangram Assembly** - Spatial composition
5. **Logic Gate Circuit** - Boolean reasoning

These tasks would significantly expand VMEvalKit's coverage of cognitive capabilities while maintaining the framework's strengths in automated, deterministic evaluation.
