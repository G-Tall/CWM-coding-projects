/*

Project euler: problem 1. All multiples of 3 and 5 below 1000.
Created by: George Tall
Email: george.tall@seh.ox.ac.uk 

*/

#include <stdio.h>

int main(){
    
    //Max number
    int N = 1000;
    int sum = 0;
    
    for(int i = 0; i < N; i++){
    	if(i % 3 == 0 || i % 5 == 0){
            sum += i;
        }
    }
    
    printf("The sum of all multiples of 3 and 5 below 1000 is %d", sum);

    return 0;
}
