%% Maximum expressiveness of the predicates
max_clauses(2).
max_vars(6).
max_body(6).

%% general
head_pred(eastbound,1).
body_pred(has_car,2).
body_pred(car_num,2).
%% payload
body_pred(has_payload3,2).
body_pred(load_num,2).
%% payload shape
body_pred(barrel3,1).
body_pred(golden_vase3,1).
body_pred(box3,1).
body_pred(diamond3,1).
body_pred(metal_pot3,1).
body_pred(oval_vase3,1).
%% colors
body_pred(car_color3,2).
body_pred(yellow3,1).
body_pred(green3,1).
body_pred(grey3,1).
body_pred(red3,1).
body_pred(blue3,1).
%% lengths
body_pred(short,1).
body_pred(long,1).
%% wheels
body_pred(has_wheel0,2).
%% roofs
body_pred(has_roof3,2).
body_pred(roof_open3,1).
body_pred(roof_foundation3,1).
body_pred(solid_roof3,1).
body_pred(braced_roof3,1).
body_pred(peaked_roof3,1).
%% walls
body_pred(braced_wall,1).
body_pred(solid_wall,1).

%% general
type(eastbound,(train,)).
type(has_car,(train,car)).
%% car number
type(car_num,(car,integer)).
%% payload
type(has_payload3,(car,load)).
type(load_num,(car,integer)).
%% payload shape
type(barrel3,(load,)).
type(golden_vase3,(load,)).
type(box3,(load,)).
type(diamond3,(load,)).
type(metal_pot3,(load,)).
type(oval_vase3,(load,)).
%% colors
type(car_color3,(car,color)).
type(yellow3,(color,)).
type(green3,(color,)).
type(grey3,(color,)).
type(red3,(color,)).
type(blue3,(color,)).
%% lengths
type(short,(car,)).
type(long,(car,)).
%% wheels
type(has_wheel0,(car,integer)).
%% roofs
type(has_roof3,(car,roof)).
%% roofs
type(roof_open3,(roof,)).
type(roof_foundation3,(roof,)).
type(solid_roof3,(roof,)).
type(braced_roof3,(roof,)).
type(peaked_roof3,(roof,)).
%% walls
type(braced_wall,(car,)).
type(solid_wall,(car,)).

%% general
direction(eastbound,(in,)).
direction(has_car,(in,out)).
%% car number
direction(car_num,(in,out)).
%% payload
direction(has_payload3,(in,out)).
direction(load_num,(in,out)).
%% payload shape
direction(barrel3,(in,)).
direction(golden_vase3,(in,)).
direction(box3,(in,)).
direction(diamond3,(in,)).
direction(metal_pot3,(in,)).
direction(oval_vase3,(in,)).
%% colors
direction(car_color3,(in,out)).
direction(yellow3,(in,)).
direction(green3,(in,)).
direction(grey3,(in,)).
direction(red3,(in,)).
direction(blue3,(in,)).
%% lengths
direction(short,(in,)).
direction(long,(in,)).
%% wheels
direction(has_wheel0,(in,out)).
%% roofs
direction(has_roof3,(in,out)).
direction(roof_open3,(in,)).
direction(roof_foundation3,(in,)).
direction(solid_roof3,(in,)).
direction(braced_roof3,(in,)).
direction(peaked_roof3,(in,)).
%% walls
direction(braced_wall,(in,)).
direction(solid_wall,(in,)).
