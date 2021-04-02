// A naive approach to kNN, running in O(mndk) time.

import java.lang.Math;
import java.util.Arrays;
import java.util.Comparator;

public class Triangle extends Base {

  private double[][] train_dist;

  public Triangle(double[][] X_train, int[] y_train) {
    super(X_train, y_train);
    // I know this is inefficient but what are you gonna do
    for(int i=0; i<X_train.length; i++) {
      for(int j=0; j<y_train.length; j++) {
        this.train_dist[i][j] = this.dist(X_train[i], X_train[j]);
      }
    }
  }

  public double[] predict(double[][] X_test, int K) {

    double[] pred = new double[0];

    for(int j=0; j<X_test.length; j++) {
      boolean[] possible = new boolean[this.N];
      for(int e=0; e<selected.length; e++) {possible[e] = true;}
      double[][] curr_neighbours = new int[K][4];

      for(int i=0; i<this.X_train.length; i++) {
        if (!possible[i]) {continue;}
        double d = this.dist(X_test[j], this.X_train[i]);

        if(i>=K) {
          for(int l=0; l<(N-(i+1)); l++) {
            if (!possible[l]) {continue;}
            if (Math.abs(d - self.train_dist[i][l] > curr_neighbours[K-1][3])) {
              possible[i + l + 1] = false;
            }
          }
        }

        for(int k=0; k<K; k++) {
          double[] tester = new double[4];
          if(Arrays.equals(curr_neighbours[k], tester) ||
             curr_neighbours[k][3] > d) {
               curr_neighbours[k][0] = i;
               curr_neighbours[k][1] = this.X_train[i];
               curr_neighbours[k][2] = this.y_train[i];
               curr_neighbours[k][3] = d;

               // this is the irritating way to delete elements of an array in Java :(

               double[][] temp = new double[(curr_neighbours.length - 1)][4];
               for(int t=0; t<curr_neighbours.length-1; t++) {
                 temp[t][0] = curr_neighbours[t][0];
                 temp[t][1] = curr_neighbours[t][1];
                 temp[t][2] = curr_neighbours[t][2];
                 temp[t][3] = curr_neighbours[t][3];
               }
               curr_neighbours = temp;
               break;
             }
        }

        // this is the irritating way to append values to an array in Java :(

        double[][] temp = new double[(neighbour_values.length + 1)][2];
        for(int t=0; t<neighbour_values.length; t++) {
          temp[t][0] = neighbour_values[t][0];
          temp[t][1] = neighbour_values[t][1];
        }
        temp[neighbour_values.length][0] = d;
        temp[neighbour_values.length][1] = this.y_train[i];
        neighbour_values = temp;
      }
      Arrays.sort(neighbour_values, Comparator.comparingDouble(o -> o[0]));

      double neighbour_sum = 0;
      for(int i=0; i<K; i++) {
        neighbour_sum += neighbour_values[i][1];
      }

      neighbour_sum /= K;
      double[] temp2 = new double[(pred.length + 1)];
      for(int t=0; t<pred.length; t++) {
        temp2[t] = pred[t];
      }
      temp2[pred.length] = neighbour_sum;
      pred = temp2;
    }
    return pred;
  }
}
