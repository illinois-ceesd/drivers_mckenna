clF = 0.004000;
clB = 0.000800;
clM = 0.000800;
clS = 0.000600;
cl0 = 0.007200;
cl1 = 0.024000;
cl2 = 0.072000;

radius = 0.50;

int_radius = 0.030;
ext_radius = 0.035;
burner_base_radius = 0.050;

burner_height = 0.10;
flame_dist = 0.0025;

mat_location = burner_height + 0.04;  //arbitrary value
mat_wide = 0.019;
mat_height = 0.054;
mat_BL_layer_x = 0.0040;
mat_BL_layer_y = 0.0040;

Point(1) = {burner_base_radius, burner_height-0.01, 0.0, clB};

Point(2) = {ext_radius, burner_height, 0.0, 0.000200};
Point(3) = {int_radius, burner_height, 0.0, 1.0};
Point(4) = {            0.005, burner_height, 0.0, 1.0};
Point(5) = {               0., burner_height, 0.0, 1.0};

delta = (mat_location - mat_BL_layer_y - burner_height - flame_dist);
flame_pos = burner_height + flame_dist;
/*Point( 8) = {     0.0,        flame_pos + 0.25*delta, 0.,      clF};*/
/*Point( 9) = {     0.0,        flame_pos + 0.75*delta, 0.,      clF};*/
Point(10) = {     0.0, mat_location - mat_BL_layer_y, 0.,      clS};
Point(11) = {     0.0, mat_location                 , 0.,      1.0};
Point(12) = {mat_wide, mat_location                 , 0.,      1.0};
Point(13) = {mat_wide, mat_location + 0.2*mat_height, 0., 0.40*clM};
Point(14) = {mat_wide, mat_location + 0.3*mat_height, 0., 0.50*clM};
Point(15) = {mat_wide, mat_location + 1.0*mat_height, 0., 1.00*clM};
Point(16) = {     0.0, mat_location + 1.0*mat_height, 0., 1.00*clM};

Point(17) = {               0.,  mat_location + 2.5*mat_height, 0., 0.5*clF};
Point(20) = {               0.,                         radius, 0., 1.0*clF};
Point(21) = {            0.120,                         radius, 0., 2.3*clF};
Point(22) = {             0.50,                         radius, 0.,     cl2};
Point(23) = {             0.50,     burner_height + flame_dist, 0.,     cl2};
Point(24) = {             0.50,                             0., 0.,     cl2};
Point(25) = {burner_base_radius,                            0., 0., 0.2*cl2};

offset = 0.0005;
Point(29) = { burner_base_radius, burner_height + flame_dist, 0.,  0.00060};
Point(30) = {ext_radius + offset, burner_height + flame_dist, 0.,  0.00240};
Point(31) = {         int_radius, burner_height + flame_dist, 0.,      clF};
Point(32) = {              0.005, burner_height + flame_dist, 0.,      1.0};
Point(33) = {                 0., burner_height + flame_dist, 0.,      clS};


/*Point(40) = {mat_wide                 , mat_location - mat_BL_layer_y, 0., 1.0};*/
Point(41) = {mat_wide + mat_BL_layer_x, mat_location - mat_BL_layer_y, 0., 1.33*clM};
/*Point(42) = {mat_wide + mat_BL_layer_x, mat_location                 , 0., 1.0};*/
Point(43) = {mat_wide + 1.35*mat_BL_layer_x, mat_location + 0.16*mat_height, 0., 1.33*clM};

Line( 1) = {2, 3};
Line( 2) = {3, 4};
Line( 3) = {3, 31};
Line( 4) = {4, 32};
Line( 5) = {4, 5};
Line( 6) = {5, 33};

Line(13) = { 1, 29};
Line(14) = {29, 30};
Line(15) = { 1,  2};
Line(16) = { 2, 30};
Line(17) = {30, 31};
Line(18) = {31, 32};
Line(19) = {32, 33};

/*Line(20) = {33,  8};*/
/*Line(21) = { 8,  9};*/
/*Line(22) = { 9, 10};*/
Line(22) = {33, 10};
Line(23) = {10, 41};
/*Line(24) = {40, 41};*/
/*Line(25) = {41, 42};*/
Line(26) = {41, 43};
Line(27) = {43, 13};
Line(28) = {13, 14};
Line(29) = {14, 15};
Line(30) = {15, 16};
Line(31) = {16, 17};
Line(32) = {17, 20};

Line(40) = {20, 21};
Line(41) = {21, 22};
Line(42) = {22, 23};
Line(43) = {23, 24};
Line(44) = {24, 25};
Line(45) = {25,  1};
Line(46) = {23, 29};

Line(50) = {10, 11};
Line(51) = {11, 12};
Line(52) = {12, 41};
/*Line(53) = {12, 42};*/
Line(54) = {12, 13};

/*Line(60) = {12, 41};*/

/*horizontal*/
Transfinite Line { 1} = 21 Using Bump 0.66;
Transfinite Line {17} = 21 Using Bump 0.66;
Transfinite Line { 2} = 77 Using Progression 1.01;
Transfinite Line {18} = 77 Using Progression 1.01;
Transfinite Line {- 5} = 12 Using Progression 1.01;
Transfinite Line {-19} = 12  Using Progression 1.01;

/*vertical*/
Transfinite Line { 3} = 38 Using Progression 1.05;
Transfinite Line {16} = 38 Using Progression 1.05;
Transfinite Line {13} = 38 Using Progression 1.0;
Transfinite Line { 4} = 38 Using Progression 1.05;
Transfinite Line { 6} = 38 Using Progression 1.05;

/*burner*/
Transfinite Line {-14} = 31 Using Progression 1.04;
Transfinite Line {-15} = 31 Using Progression 1.06;

/*material*/
/*Transfinite Line {25} = 16 Using Progression 1.01;*/
Transfinite Line {-50} = 17 Using Progression 1.05;
Transfinite Line {52} = 17 Using Progression 1.05;
Transfinite Line {-27} = 17 Using Progression 1.0;

Transfinite Line {-23} = 41 Using Progression 1.0;
Transfinite Line {-51} = 41 Using Progression 1.01;

/*Transfinite Line {24} = 13 Using Progression 1.00;*/
/*Transfinite Line {-27} = 13 Using Progression 1.04;*/
/*Transfinite Line {53} = 13 Using Progression 1.066;*/

Transfinite Line {26} = 26 Using Progression 1.005;
Transfinite Line {54} = 26 Using Progression 1.01;

/*Transfinite Line {-50} = 12 Using Progression 1.06;*/
/*Transfinite Line {-27} = 12 Using Progression 1.06;*/
/*Transfinite Line { 60} = 12 Using Progression 1.06;*/

Line Loop(40) = {43,44,45,13,-46};
Plane Surface(111) = {40};

Line Loop(41) = {14,17:19,22,23,26:32,40:42,46};
Plane Surface(1) = {41};

Point(49) = {   0.90*ext_radius, flame_pos + 1.25*delta, 0., 0.40*clF};
Point(50) = {   0.75*ext_radius, flame_pos + 0.75*delta, 0., 0.40*clF};
Point(51) = {   0.50*ext_radius, flame_pos + 0.75*delta, 0., 0.40*clF};
Point(52) = {   0.25*ext_radius, flame_pos + 0.75*delta, 0., 0.40*clF};
Point(53) = {   0.25*int_radius, flame_pos + 0.25*delta, 0., 0.40*clF};
Point(54) = {   0.50*int_radius, flame_pos + 0.25*delta, 0., 0.45*clF};
Point(55) = {   0.75*int_radius, flame_pos + 0.25*delta, 0., 0.50*clF};
Point(56) = {   1.00*int_radius, flame_pos + 0.25*delta, 0., 0.50*clF};
Point(57) = {   1.25*int_radius, flame_pos + 0.25*delta, 0., 0.55*clF};
Point(58) = {   1.50*int_radius, flame_pos + 0.25*delta, 0., 0.60*clF};
Point(59) = {   1.75*int_radius, flame_pos + 0.25*delta, 0., 0.65*clF};
Point{49:59} In Surface{1};

Point(60) = {  0.060, mat_location - 0.57*mat_height, 0., 0.7*clF};
Point(61) = {  0.070, mat_location - 0.57*mat_height, 0., 0.8*clF};
/*Point(62) = {  0.080, mat_location - 0.57*mat_height, 0., 1.00*clF};*/
/*Point(63) = {  0.090, mat_location - 0.57*mat_height, 0., 1.00*clF};*/
Point{60:61} In Surface{1};

/*Point(70) = {  0.080, mat_location - 0.57*mat_height, 0., 1.00*clF};*/
Point(71) = {  0.080, mat_location - 0.55*mat_height, 0., 0.90*clF};
Point(72) = {  0.090, mat_location - 0.50*mat_height, 0., 1.00*clF};
Point(73) = {  0.098, mat_location - 0.40*mat_height, 0., 1.00*clF};
Point(74) = {  0.106, mat_location - 0.30*mat_height, 0., 1.00*clF};
Point(75) = {  0.114, mat_location + 0.00*mat_height, 0., 1.05*clF};
Point(76) = {  0.118, mat_location + 0.33*mat_height, 0., 1.10*clF};
Point(77) = {  0.120, mat_location + 0.66*mat_height, 0., 1.15*clF};
Point(78) = {  0.120, mat_location + 1.00*mat_height, 0., 1.20*clF};
Point(79) = {  0.120, mat_location + 1.33*mat_height, 0., 1.25*clF};
Point(80) = {  0.120, mat_location + 1.66*mat_height, 0., 1.30*clF};
Point(81) = {  0.120, mat_location + 2.00*mat_height, 0., 1.35*clF};
Point(82) = {  0.120, mat_location + 2.50*mat_height, 0., 1.40*clF};
Point(83) = {  0.120, mat_location + 3.00*mat_height, 0., 1.45*clF};
Point(84) = {  0.120, mat_location + 3.50*mat_height, 0., 1.50*clF};
Point(85) = {  0.120, mat_location + 4.00*mat_height, 0., 1.60*clF};
Point(86) = {  0.120, mat_location + 4.50*mat_height, 0., 1.70*clF};
Point(87) = {  0.120, mat_location + 5.00*mat_height, 0., 1.80*clF};
Point(88) = {  0.120, mat_location + 5.50*mat_height, 0., 1.90*clF};
Point(89) = {  0.120, mat_location + 6.00*mat_height, 0., 2.00*clF};
/*Point(90) = {  0.120, mat_location + 7.00*mat_height, 0., 2.10*clF};*/
/*Point(91) = {  0.120, mat_location + 8.00*mat_height, 0., 2.20*clF};*/
/*Point(92) = {  0.120, mat_location + 9.00*mat_height, 0., 2.30*clF};*/
Point{71:89} In Surface{1};


Point(101) = {              0.00050, flame_dist + burner_height, 0., 1.0};
Point(102) = {int_radius + 0.000200,              burner_height, 0., 1.0};
Point(103) = {int_radius - 0.000200,              burner_height, 0., 1.0};
Point(104) = {int_radius           ,   0.000025 + burner_height, 0., 1.0};
Point(105) = {      0.005 - 0.00050, flame_dist + burner_height, 0., 1.0};
Point(106) = {      0.005 + 0.00050, flame_dist + burner_height, 0., 1.0};
Point(107) = {        0.           ,  -0.000120 + mat_location , 0., 1.0};

Line Loop(42) = {1,3,-17,-16};
Plane Surface(2) = {42};
Transfinite Surface {2};

Line Loop(43) = {2,4,-18,-3};
Plane Surface(3) = {43};
Transfinite Surface {3};

Line Loop(44) = {-4,5,6,-19};
Plane Surface(4) = {44};
Transfinite Surface {4};

Line Loop(45) = {15,16,-14,-13};
Plane Surface(5) = {45};
Transfinite Surface {5};

Line Loop(46) = {51,52,-23,50};
Plane Surface(6) = {46};
Transfinite Surface {6};

/*Line Loop(47) = {24,25,-53,52};*/
/*Plane Surface(7) = {-47};*/
/*Transfinite Surface {7};*/

/*Line Loop(48) = {-26,-53,54,-27};*/
/*Plane Surface(8) = {48};*/
/*Transfinite Surface {8};*/

Line Loop(47) = {27, -54, 52, 26};
Plane Surface(7) = {-47};
Transfinite Surface {7};

/*#########################*/


//Point(80) = {0.0, mat_location + mat_BL_layer_y, 0., 0.25*clM};//
//Point(81) = {mat_wide - mat_BL_layer_x, mat_location + mat_BL_layer_y, 0., clM};
//Point(82) = {mat_wide - mat_BL_layer_x, mat_location + 0.2*mat_height, 0., clM};
//Point(83) = {0.0, mat_location + 0.3*mat_height, 0., clM};

// Line(60) = {11, 80};
// /*Line(61) = {12, 81};*/
// /*Line(62) = {13, 82};*/
// Line(63) = {16, 83};
// Line(64) = {83, 80};
// /*Line(65) = {80, 81};*/
// /*Line(66) = {81, 82};*/
// Line(67) = {83, 14};

// /*Transfinite Line {60} = 16 Using Progression 1.066;*/
// /*Transfinite Line {61} = 16 Using Progression 1.066;*/
// /*Transfinite Line {62} = 16 Using Progression 1.066;*/
// /*Transfinite Line {-65} = 51 Using Progression 1.025;*/
// /*Transfinite Line {66} = 41 Using Progression 1.02;*/

// /*Line Loop(50) = {51,61,-65,-60};*/
// /*Plane Surface(10) = {-50};*/
// /*Transfinite Surface {10};*/

// /*Line Loop(51) = {54,62,-66,-61};*/
// /*Plane Surface(11) = {-51};*/
// /*Transfinite Surface {11};*/

// /*Line Loop(68) = {28,-67,64,65,66,-62};*/
// /*Plane Surface(12) = {-68};*/

// Line Loop(68) = {51,54,28,-67,64,-60};
// Plane Surface(12) = {-68};

// Line Loop(69) = {29,30,63,67};
// Plane Surface(13) = {-69};

/*#########################*/

Physical Line("inlet") = {1,2,5};
Physical Line("symmetry") = {6,22,50,31,32};
Physical Line("burner") = {45,15};
Physical Line("solid") = {51,54,28,29,30};
Physical Line("outlet") = {40,41};
Physical Line("linear") = {42,43,44};
//Physical Line("wall_sym") = {60,63,64};
Physical Surface("fluid") = {1:7,111};
// Physical Surface("wall_insert") = {12};
// Physical Surface("wall_surround") = {13};

/*Coherence;*/
