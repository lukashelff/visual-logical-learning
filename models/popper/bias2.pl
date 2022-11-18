
max_clauses(3).
max_vars(5).
max_body(6).

%% general
head_pred(f,1).
body_pred(has_car,2).
body_pred(has_load,2).
body_pred(behind,2).
%% lengths
body_pred(short,1).
body_pred(long,1).
%% wheels
body_pred(two_wheels,1).
body_pred(three_wheels,1).
%% roofs
body_pred(roof_open,1).
body_pred(roof_closed,1).
%% payloads
body_pred(zero_load,1).
body_pred(barrel,1).
body_pred(golden_vase,1).
body_pred(box,1).
body_pred(diamond,1).
body_pred(metal_pot,1).
body_pred(oval_vase,1).


%% general
type(f,(train,)).
type(has_car,(train,car)).
type(has_load,(car,load)).
type(behind,(car,car)).
%% lengths
type(short,(car,)).
type(long,(car,)).
%% wheels
type(two_wheels,(car,)).
type(three_wheels,(car,)).
%% roofs
type(roof_open,(car,)).
type(roof_closed,(car,)).
%% payloads
type(zero_load,(car,)).
type(barrel,(load,)).
type(golden_vase,(load,)).
type(box,(load,)).
type(diamond,(load,)).
type(metal_pot,(load,)).
type(oval_vase,(load,)).

%% general
direction(f,(in,)).
direction(has_car,(in,out)).
direction(has_load,(in,out)).
direction(behind,(in,out)).
%% lengths
direction(short,(in,)).
direction(long,(in,)).
%% wheels
direction(two_wheels,(in,)).
direction(three_wheels,(in,)).
%% roofs
direction(roof_open,(in,)).
direction(roof_closed,(in,)).
%% payloads
direction(zero_load,(in,)).
direction(barrel,(in,)).
direction(golden_vase,(in,)).
direction(box,(in,)).
direction(diamond,(in,)).
direction(metal_pot,(in,)).
direction(oval_vase,(in,)).