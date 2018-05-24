#include <stdio.h>

int main(){
    
    int N = 4000000;
    int sum = 2;
    int previous = 1;
    int previous_stored;
    int next = 2;

    while(next < N){
        previous_stored = next;
 
        next = next + previous;
        
        if(next > N){ 
              break;
        }
        
        previous = previous_stored;
        printf("%d, \n", next);
	
        if(next % 2 == 0){
            sum += next;
            printf("%d! \n", sum);
        }
    }

    printf("sum is %d", sum);    

    return 0;
}
