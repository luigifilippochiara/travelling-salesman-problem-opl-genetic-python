/*********************************************
 * OPL 12.8.0.0 Model
 * Author: damnko
 * Creation Date: Nov 23, 2018 at 3:46:40 PM
 *********************************************/

 // Params
 int totalHoles = ...;
 setof(int) I = asSet(1..totalHoles); // row index
 float C[I][I] = ...;
 int zeroHoleID = ...;
 float timeLimit = 9999; // no time limit by default
 
 // Decision variables
 dvar int+ x[I][I];
 dvar boolean y[I][I];
 
 // Measure optimization time and iteratively set time limit param
 // which will be passed from from command line
 float temp;
 execute{
  var before = new Date();
  temp = before.getTime();
  // set time limit
  cplex.tilim=timeLimit;
 }
 
 // Model
 minimize sum(i in I, j in I) C[i][j]*y[i][j];
 
 subject to{
 totalFlow: sum(j in I) x[zeroHoleID][j] == card(I); 
 
 forall(k in I : k != zeroHoleID){
 netFlow: sum(i in I)x[i][k] - sum(j in I)x[k][j] == 1; 
 }
 
 forall(i in I){
 outBound: sum(j in I)y[i][j] == 1;  
 }
 forall(j in I){
 inBound: sum(i in I)y[i][j] == 1; 
 }
 
 forall(i in I, j in I){
 flowUsage: x[i][j] <= y[i][j]*card(I); 
 }
 
 }
 
 // Execute after optimization is completed
 execute{ 
  // Output solving time
  var after = new Date();
  writeln(after);
  writeln("solving time ~= ",after.getTime()-temp); 
 
  // Export results in txt file
  writeln('Done');
  writeln(x);
  var f=new IloOplOutputFile("export.txt");
  f.writeln(x);
  f.close();
 }
 