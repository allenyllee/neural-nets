#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "./neural.h"
#include "./model.h"
#include "./data.h"

#define DATASIZE 6

int main(){
    int count = 0;
    Unit a[PAR_NUM];
    Data d[DATASIZE] = {{.x[0]=1.2, .x[1]=0.7, .label=1},
                {.x[0]=-0.3, .x[1]=0.5, .label=-1},
                {.x[0]=-3, .x[1]=-1, .label=1},
                {.x[0]=0.1, .x[1]=1.0, .label=-1},
                {.x[0]=3.0, .x[1]=1.1, .label=-1},
                {.x[0]=2.1, .x[1]=-3, .label=1}};

    for(int i=0; i<PAR_NUM; i++){
        a[i].value = 1;
        a[i].grad = 0;
    }

    int j=0;
    int i=0;
    srand(time(NULL));

    while (1){
        printf("loop %d\n",j);
        i = rand() % DATASIZE ;
        if ( model1(a,&d[i],1) * d[i].label <0 ){
            count = 0;
        }else {
            count += 1;
            if(count == DATASIZE*5) break;
        }
        j++;
    }

    printf("=============\n");

    for (int i=0; i<DATASIZE; i++){
        model1(a,&d[i],0);
    }

    return 0;
}