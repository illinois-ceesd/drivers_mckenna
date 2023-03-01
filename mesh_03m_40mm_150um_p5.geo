clF = 0.010000;
clB = 0.002000;
clM = 0.002000;
clS = 0.001200;
cl0 = 0.048000;
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
Point(4) = {            0.010, burner_height, 0.0, 1.0};
Point(5) = {               0., burner_height, 0.0, 1.0};

delta = (mat_location - mat_BL_layer_y - burner_height - flame_dist);
flame_pos = burner_height + flame_dist;

Point(10) = {     0.0, mat_location - mat_BL_layer_y, 0.,  clS};
Point(11) = {     0.0, mat_location                 , 0., 1.00};
Point(12) = {mat_wide, mat_location                 , 0., 1.00};
Point(13) = {mat_wide, mat_location + 0.2*mat_height, 0., 1.00};

Point(15) = {mat_wide, mat_location + 1.0*mat_height, 0., 1.00};
Point(16) = {     0.0, mat_location + 1.0*mat_height, 0., 1.00};
Point(17) = {     0.0, mat_location + 1.0*mat_height + mat_BL_layer_y, 0., 1.00*clS};

Point(20) = {               0.,                         radius, 0., 0.33*cl2};
Point(21) = {             0.12,                         radius, 0., 0.66*cl2};
Point(22) = {             0.50,                         radius, 0.,     cl2};
Point(24) = {             0.50,                             0., 0.,     cl2};
Point(25) = {burner_base_radius,                            0., 0., 6.66*clB};

offset = 0.0005;
Point(29) = { burner_base_radius, burner_height + flame_dist, 0.,  1.0};
Point(30) = {ext_radius + offset, burner_height + flame_dist, 0.,  1.0};
Point(31) = {         int_radius, burner_height + flame_dist, 0.,      1.0};
Point(32) = {              0.010, burner_height + flame_dist, 0.,      1.0};
Point(33) = {                 0., burner_height + flame_dist, 0.,      1.0};


Point(41) = {mat_wide + mat_BL_layer_x, mat_location - mat_BL_layer_y, 0., 1.0};
Point(43) = {mat_wide + mat_BL_layer_x, mat_location + 0.2*mat_height, 0., 1.0};
Point(45) = {mat_wide + mat_BL_layer_x, mat_location + 1.0*mat_height + mat_BL_layer_y, 0., 1.0};

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

Line(22) = {33, 10};
Line(23) = {10, 41};
Line(26) = {41, 43};
Line(27) = {43, 45};
Line(28) = {45, 17};

Line(32) = {17, 20};

Line(40) = {20, 21};
Line(41) = {21, 22};
Line(42) = {22, 24};
Line(43) = {24, 25};
Line(44) = {25,  1};

Line(50) = {10, 11};
Line(51) = {11, 12};
Line(52) = {12, 41};
Line(53) = {13, 43};
Line(54) = {12, 13};
Line(55) = {13, 15};
Line(56) = {15, 16};
Line(57) = {45, 15};
Line(58) = {16, 17};


Transfinite Line {22} = 26 Using Bump 0.8;

/*horizontal*/
Transfinite Line { 1} = 12 Using Bump 0.8;
Transfinite Line {17} = 12 Using Bump 0.66;
Transfinite Line { 2} = 30 Using Progression 1.04;
Transfinite Line {18} = 30 Using Progression 1.04;
Transfinite Line {- 5} = 11 Using Progression 1.04;
Transfinite Line {-19} = 11  Using Progression 1.04;

/*burner*/
Transfinite Line {-14} = 25 Using Progression 1.04;
Transfinite Line {-15} = 25 Using Progression 1.06;

/*vertical*/
Transfinite Line { 3} = 15 Using Progression 1.05;
Transfinite Line {16} = 15 Using Progression 1.05;
Transfinite Line {13} = 15 Using Progression 1.0;
Transfinite Line { 4} = 15 Using Progression 1.05;
Transfinite Line { 6} = 15 Using Progression 1.05;


/*material*/
Transfinite Line {-50} = 7 Using Progression 1.2;
Transfinite Line { 52} = 7 Using Progression 1.2;
Transfinite Line { 53} = 7 Using Progression 1.2;
Transfinite Line {-57} = 7 Using Progression 1.2;
Transfinite Line { 58} = 7 Using Progression 1.2;
Transfinite Line { 55} = 31 Using Progression 1.0;
Transfinite Line { 27} = 31 Using Progression 1.0;
Transfinite Line { 56} = 16 Using Progression 1.0;
Transfinite Line { 28} = 16 Using Progression 1.0;

Transfinite Line {-23} = 16 Using Progression 1.01;
Transfinite Line {-51} = 16 Using Progression 1.03;

Transfinite Line {26} = 11 Using Progression 1.005;
Transfinite Line {54} = 11 Using Progression 1.01;

Line Loop(41) = {13,14,17:19,22,23,26:28,32,40:44};
Plane Surface(1) = {41};


Point(49) = {   1.00*ext_radius, flame_pos + 1.25*delta, 0., 0.40*clF};
Point(50) = {   1.00*ext_radius, flame_pos + 0.75*delta, 0., 0.40*clF};
Point(51) = {   0.66*ext_radius, flame_pos + 0.75*delta, 0., 0.40*clF};
Point(52) = {   0.33*ext_radius, flame_pos + 0.75*delta, 0., 0.40*clF};
Point(53) = {   0.50*int_radius, flame_pos + 0.25*delta, 0., 0.40*clF};
Point(54) = {   1.00*int_radius, flame_pos + 0.25*delta, 0., 0.40*clF};
Point(55) = {   1.50*int_radius, flame_pos + 0.25*delta, 0., 0.40*clF};
Point{49:55} In Surface{1};


Point(61) = {  0.070, mat_location - 0.7*mat_height, 0., 0.5*clF};
Point{61} In Surface{1};

Point(71) = {  0.10, mat_location - 0.50*mat_height, 0., 0.7*clF};
Point(72) = {  0.12, mat_location + 0.00*mat_height, 0., 0.9*clF};
Point(73) = {  0.13, mat_location + 0.66*mat_height, 0., 1.0*clF};
Point(74) = {  0.13, mat_location + 1.33*mat_height, 0., 1.0*clF};
Point(75) = {  0.13, mat_location + 2.00*mat_height, 0., 1.0*clF};
Point{71:75} In Surface{1};

Point(91) = {int_radius + 0.000500,   0.000125 + burner_height, 0., 1.0};
Point(92) = {int_radius - 0.000500,   0.000125 + burner_height, 0., 1.0};
Point(93) = {             0.002500, flame_dist + burner_height, 0., 1.0};
Point(94) = {     0.010 - 0.002500, flame_dist + burner_height, 0., 1.0};
Point(95) = {     0.010 + 0.002500, flame_dist + burner_height, 0., 1.0};
Point(96) = {        0.           ,  -0.000300 + mat_location , 0., 1.0};

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

Line Loop(47) = {-53, -54, 52, 26};
Plane Surface(7) = {-47};
Transfinite Surface {7};

Line Loop(48) = {-53, 55, -57, -27};
Plane Surface(8) = {48};
Transfinite Surface {8};

Line Loop(49) = {58, -28, 57, 56};
Plane Surface(9) = {49};
Transfinite Surface {9};

/*#########################*/

Physical Line("inlet") = {1,2,5};
Physical Line("symmetry") = {6,22,50,32,58};
Physical Line("burner") = {44,15};
Physical Line("solid") = {51,54,55,56};
Physical Line("outlet") = {40,41};
Physical Line("linear") = {42,43};
Physical Surface("fluid") = {1:9};

//+
Show "*";

