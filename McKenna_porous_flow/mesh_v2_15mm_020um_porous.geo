clF = 0.001600*1.4142*1.4142;
clB = 0.000800*1.4142;
clM = 0.000800*1.4142*1.4142;
clS = 0.000600*1.4142;
cl0 = 0.007200*1.4142;
cl1 = 0.024000*1.4142;
cl2 = 0.072000*1.4142;

radius = 1.0;

int_radius = 2.38*25.4/2000;
ext_radius = 2.89*25.4/2000;
burner_base_radius = 4.725*25.4/2000;

burner_height = 0.10;
flame_dist = 0.0025;

mat_location = burner_height + 0.025;  //arbitrary value
mat_wide = 0.5*1.25*25.4/1000;
mat_thick = 0.75*25.4/1000;
insulator = 1.00*25.4/1000 + mat_thick;
insulator_wide = 0.5*1.00*25.4/1000;
mat_holder = 1.75*25.4/1000 + mat_thick;
mat_BL_layer_x = 0.0030;
mat_BL_layer_y = 0.0030;


Point(2) = {ext_radius, burner_height, 0.0, 0.000200};
Point(3) = {int_radius, burner_height, 0.0, 1.0};
Point(4) = {            0.005, burner_height, 0.0, 1.0};
Point(5) = {               0., burner_height, 0.0, 1.0};

delta = (mat_location - mat_BL_layer_y - burner_height - flame_dist);
flame_pos = burner_height + flame_dist;

Point( 8) = {     0.0,        flame_pos + 0.33*delta, 0.,     clS};
Point( 9) = {     0.0,        flame_pos + 0.66*delta, 0.,     clS};
Point(10) = {     0.0, mat_location - mat_BL_layer_y, 0.,     clS};
Point(11) = {     0.0, mat_location                 , 0., 0.7*clS};
Point(12) = {mat_wide, mat_location                 , 0.,     1.0};
Point(13) = {mat_wide,      mat_location + mat_thick, 0.,     1.0};
Point(14) = {mat_wide,      mat_location + insulator, 0.,     clM};

Point(15) = {mat_wide, mat_location + 1.0*mat_holder, 0., 1.0};
Point(16) = {     0.0, mat_location + 1.0*mat_holder, 0., clM};
Point(17) = {     0.0, mat_location + 1.0*mat_holder + mat_BL_layer_y, 0., 1.00*clS};
Point(18) = {     0.0,                          burner_height + 0.550, 0., 6.00*clF};

Point(20) = {              0.00, radius, 0., 12.5*clF};
Point(21) = {              0.25, radius, 0., 12.5*clF};
Point(22) = {            radius, radius, 0.,      cl2};
Point(24) = {            radius,     0., 0.,      cl2};
Point(25) = {burner_base_radius,     0., 0., 6.66*clB};

aux = 0.5*(burner_base_radius + ext_radius);
Point(26) = {burner_base_radius, burner_height-0.0110, 0.0, clB};
Point(27) = {               aux, burner_height-0.0055, 0.0, clB};

offset = 0.0;
Point(28) = { burner_base_radius, burner_height + flame_dist, 0., 1.0};
Point(29) = {                aux, burner_height + flame_dist, 0., clB};
Point(30) = {ext_radius + offset, burner_height + flame_dist, 0., 1.0};
Point(31) = {         int_radius, burner_height + flame_dist, 0., 1.0};
Point(32) = {              0.005, burner_height + flame_dist, 0., 1.0};
Point(33) = {                 0., burner_height + flame_dist, 0., clS};


Point(41) = {mat_wide + mat_BL_layer_x, mat_location - mat_BL_layer_y, 0., 1.0};
Point(43) = {mat_wide + mat_BL_layer_x, mat_location + mat_thick, 0., 1.0};
Point(44) = {mat_wide + mat_BL_layer_x, mat_location + insulator, 0., 1.0};
Point(45) = {mat_wide + mat_BL_layer_x, mat_location + 1.0*mat_holder + mat_BL_layer_y, 0., 1.0};

Line( 1) = {2, 3};
Line( 2) = {3, 4};
Line( 3) = {3, 31};
Line( 4) = {4, 32};
Line( 5) = {4, 5};
Line( 6) = {5, 33};

Line(10) = {26, 28};
Line(11) = {28, 29};
Line(12) = {29, 30};
Line(13) = {27, 29};
Line(14) = {26, 27};
Line(15) = {27,  2};
Line(16) = { 2, 30};
Line(17) = {30, 31};
Line(18) = {31, 32};
Line(19) = {32, 33};

Line(20) = {33,  8};
Line(21) = { 8,  9};
Line(22) = { 9, 10};
Line(23) = {10, 41};
Line(26) = {41, 43};
Line(27) = {43, 44};
Line(28) = {44, 45};
Line(29) = {45, 17};

Line(31) = {17, 18};
Line(32) = {18, 20};

Line(40) = {20, 21};
Line(41) = {21, 22};
Line(42) = {22, 24};
Line(43) = {24, 25};
Line(44) = {25, 26};

Line(50) = {10, 11};
Line(51) = {11, 12};
Line(52) = {12, 41};
Line(53) = {13, 43};
Line(54) = {12, 13};
Line(55) = {13, 14};
Line(56) = {14, 15};
Line(57) = {15, 16};
Line(58) = {45, 15};
Line(59) = {16, 17};
Line(60) = {44, 14};

delta_mat = mat_wide - insulator_wide;
Point(51) = {insulator_wide, mat_location + insulator, 0., 0.9*clM};
Point(52) = {           0.0, mat_location + insulator, 0., 1.0*clM};
Point(53) = {           0.0, mat_location + mat_thick, 0., 0.6*clM};
Point(54) = {insulator_wide, mat_location + mat_thick, 0., 0.3*clM};
Point(55) = {insulator_wide, mat_location + delta_mat, 0., 0.3*clM};
Point(56) = {           0.0, mat_location + delta_mat, 0., 0.3*clM};

Line(61) = {11, 56};
/*Line(62) = {14, 51};*/
Line(63) = {56, 53};
Line(64) = {54, 13};
Line(65) = {16, 52};
Line(66) = {51, 52};
Line(67) = {52, 53};
Line(68) = {53, 54};
Line(69) = {54, 51};
Line(70) = {54, 55};
Line(71) = {55, 56};
Line(72) = {55, 12};

/*Transfinite Line {21} = 16 Using Bump 0.6;*/

/*horizontal*/
Transfinite Line {  1} = 19 Using Bump 0.75;
Transfinite Line { 17} = 19 Using Bump 0.75;
Transfinite Line {  2} = 61 Using Progression 1.01;
Transfinite Line { 18} = 61 Using Progression 1.01;
Transfinite Line {- 5} = 10 Using Progression 1.01;
Transfinite Line {-19} = 10  Using Progression 1.01;

/*vertical*/
Transfinite Line {  3} = 27 Using Progression 1.1;
Transfinite Line { 16} = 27 Using Progression 1.1;
Transfinite Line {  4} = 27 Using Progression 1.1;
Transfinite Line {  6} = 27 Using Progression 1.1;
Transfinite Line { 13} = 27 Using Progression 1.0;
Transfinite Line { 10} = 27 Using Progression 1.0;

/*burner*/
Transfinite Line {-15} = 21 Using Progression 1.055;
Transfinite Line {-12} = 21 Using Progression 1.055;
Transfinite Line { 14} = 11 Using Progression 1.00;
Transfinite Line { 11} = 11 Using Progression 1.00;

/*material*/
Transfinite Line {-50} = 13 Using Progression 1.100;
Transfinite Line { 52} = 13 Using Progression 1.100;
Transfinite Line { 53} = 13 Using Progression 1.100;
Transfinite Line {-58} = 13 Using Progression 1.100;
Transfinite Line { 59} = 13 Using Progression 1.100;
Transfinite Line {-60} = 13 Using Progression 1.100;

Transfinite Line { 26} = 41 Using Progression 1.000;
Transfinite Line { 54} = 41 Using Progression 1.005;
Transfinite Line { 70} = 41 Using Progression 1.0;

Transfinite Line { 23} = 35 Using Progression 1.002;
Transfinite Line {-51} = 35 Using Progression 1.010;
Transfinite Line {-71} = 35 Using Progression 1.0;

Transfinite Line { 27} = 26 Using Progression 1.04;
Transfinite Line { 55} = 26 Using Progression 1.04;

Transfinite Line {-28} = 16 Using Progression 1.005;
Transfinite Line {-56} = 16 Using Progression 1.02;

Transfinite Line { 57} = 19 Using Progression 1.0;
Transfinite Line {-29} = 19 Using Progression 1.02;

Transfinite Line {-64} = 13 Using Progression 1.1;
Transfinite Line {-72} = 13 Using Progression 1.1;
Transfinite Line { 61} = 13 Using Progression 1.1;

/*#############################################*/

/*Fluid*/
Line Loop(1) = {10:12,17:23,26:29,31,32,40:44};
Plane Surface(1) = {1};

Line Loop(2) = {1,3,-17,-16};
Plane Surface(2) = {2};
Transfinite Surface {2} Alternate;

Line Loop(3) = {2,4,-18,-3};
Plane Surface(3) = {3};
Transfinite Surface {3} Alternate;

Line Loop(4) = {-4,5,6,-19};
Plane Surface(4) = {4};
Transfinite Surface {4} Alternate;

Line Loop(5) = {15,16,-12,-13};
Plane Surface(5) = {5};
Transfinite Surface {5} Alternate;

Line Loop(6) = {14,13,-11,-10};
Plane Surface(6) = {6};
Transfinite Surface {6} Alternate;

/*Material BL*/
Line Loop(11) = {51,52,-23,50};
Plane Surface(11) = {11};
Transfinite Surface {11};

Line Loop(12) = {-53, -54, 52, 26};
Plane Surface(12) = {-12};
Transfinite Surface {12};

Line Loop(13) = {-60,-27, -53, 55};
Plane Surface(13) = {13};
Transfinite Surface {13};

Line Loop(14) = {60, 56, -58, -28};
Plane Surface(14) = {14};
Transfinite Surface {14} Right;

Line Loop(15) = {59,-29, 58, 57};
Plane Surface(15) = {15};
Transfinite Surface {15};

/*Sample*/
Line Loop(16) = {70,71,63,68};
Plane Surface(16) = {16};

Line Loop(17) = {70,72,54,-64};
Plane Surface(17) = {-17};
Transfinite Surface {17} Left;

Line Loop(18) = {72,-51,61,-71};
Plane Surface(18) = {18};
Transfinite Surface {18} Right;

/*Alumina*/
Line Loop(20) = {66,67,68,69};
Plane Surface(20) = {-20};

/*Graphite*/
Line Loop(21) = {56,57,65,-66,-69,64,55};
Plane Surface(21) = {-21};

/*#########################*/




/*Point(49) = {   0.90*ext_radius, flame_pos + 1.25*delta, 0., 0.40*clF};*/
/*Point(57) = {   1.00*int_radius, flame_pos + 1.25*delta, 0., 1.00*clF};*/
/*Point(58) = {   1.00*int_radius, flame_pos + 1.00*delta, 0., 1.00*clF};*/
/*Point(59) = {   1.00*int_radius, flame_pos + 0.75*delta, 0., 1.00*clF};*/
/*Point(60) = {   0.75*int_radius, flame_pos + 0.75*delta, 0., 1.00*clF};*/
/*Point(61) = {   0.50*int_radius, flame_pos + 0.75*delta, 0., 1.00*clF};*/
/*Point(62) = {   0.25*int_radius, flame_pos + 0.75*delta, 0., 1.00*clF};*/
/*Point(63) = {   0.25*int_radius, flame_pos + 0.25*delta, 0., 1.00*clF};*/
/*Point(64) = {   0.50*int_radius, flame_pos + 0.25*delta, 0., 1.00*clF};*/
/*Point(65) = {   0.75*int_radius, flame_pos + 0.25*delta, 0., 1.00*clF};*/
/*Point(66) = {   1.00*int_radius, flame_pos + 0.25*delta, 0., 1.00*clF};*/
/*Point(67) = {   1.25*int_radius, flame_pos + 0.25*delta, 0., 1.00*clF};*/
/*Point(68) = {   1.50*int_radius, flame_pos + 0.25*delta, 0., 1.00*clF};*/
/*Point(69) = {   1.75*int_radius, flame_pos + 0.25*delta, 0., 1.00*clF};*/
/*Point{59:69} In Surface{1};*/

Point(70) = {  0.080, burner_height - 0.028, 0., 2.250*clF};
Point(71) = {  0.120, burner_height - 0.025, 0., 2.500*clF};
Point(72) = {  0.160, burner_height - 0.020, 0., 2.625*clF};
Point(73) = {  0.200, burner_height - 0.012, 0., 2.750*clF};
Point(74) = {  0.230, burner_height + 0.000, 0., 2.875*clF};
Point(75) = {  0.245, burner_height + 0.020, 0., 3.000*clF};
Point(76) = {  0.250, burner_height + 0.050, 0., 3.125*clF};
Point(77) = {  0.250, burner_height + 0.080, 0., 3.250*clF};
Point(78) = {  0.250, burner_height + 0.120, 0., 3.375*clF};
Point(79) = {  0.250, burner_height + 0.160, 0., 3.500*clF};
Point(80) = {  0.250, burner_height + 0.200, 0., 3.625*clF};
Point(81) = {  0.250, burner_height + 0.240, 0., 3.750*clF};
Point(82) = {  0.250, burner_height + 0.280, 0., 4.000*clF};
Point(83) = {  0.250, burner_height + 0.320, 0., 4.250*clF};
Point(84) = {  0.250, burner_height + 0.360, 0., 4.500*clF};
Point(85) = {  0.250, burner_height + 0.400, 0., 4.750*clF};
Point(86) = {  0.250, burner_height + 0.450, 0., 5.000*clF};
Point(87) = {  0.250, burner_height + 0.500, 0., 5.500*clF};
Point(88) = {  0.250, burner_height + 0.550, 0., 6.000*clF};
Point{70:88} In Surface{1};

Point(90) = {    0.0050, flame_pos + 0.33*delta, 0., 1.5*clS};
Point(91) = {    0.0175, flame_pos + 0.33*delta, 0., 1.5*clS};
Point(92) = {int_radius, flame_pos + 0.33*delta, 0., 1.5*clS};
Point(93) = {    0.0175, flame_pos + 0.66*delta, 0., 1.5*clS};
Point(94) = {    0.0050, flame_pos + 0.66*delta, 0., 1.5*clS};
Point(95) = {int_radius, flame_pos + 0.66*delta, 0., 1.5*clS};
Point(96) = {int_radius, flame_pos + 1.00*delta, 0., 1.5*clS};
Point(97) = {int_radius, flame_pos + 1.33*delta, 0., 1.5*clS};
Point{90:97} In Surface{1};

Point(100) = {0.1, 0.1, 0., 1.0};
Point(101) = {             0.000500, flame_dist + burner_height, 0., 1.0};
Point(102) = {int_radius + 0.000281,              burner_height, 0., 1.0};
Point(103) = {int_radius - 0.000281,              burner_height, 0., 1.0};
Point(104) = {int_radius           ,   0.000020 + burner_height, 0., 1.0};
Point(105) = {      0.005 - 0.00050, flame_dist + burner_height, 0., 1.0};
Point(106) = {      0.005 + 0.00050, flame_dist + burner_height, 0., 1.0};
Point(107) = {        0.           ,  -0.000120 + mat_location , 0., 1.0};



Physical Line("inlet") = {1,2,5};
Physical Line("symmetry") = {6,20:22,50,31,32,59,61,63};
Physical Line("burner") = {44,14,15};
/*Physical Line("outlet") = {40};*/
Physical Line("linear") = {40,41,42,43};
Physical Line("wall_sym") = {65,67};
/*Physical Line("interface") = {64,68};*/
Physical Surface("fluid") = {1:6,11:15};
Physical Surface("sample") = {16:18};
Physical Surface("wall_alumina") = {20};
Physical Surface("wall_graphite") = {21};

//+
Show "*";
