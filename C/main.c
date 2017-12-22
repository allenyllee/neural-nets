#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "neural.h"
#include "model.h"
#include "data.h"

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


    Model1 m;
    m.init = _init;


    int j=0;
    int i=0;
    srand(time(NULL));
/*
    while (1){
        printf("loop %d\n",j);
        //i = rand() % DATASIZE ;
        i = j % DATASIZE ;
        m.init(&m, a, &d[i]);

        if ( m.forward(&m) * m.data->label <0 ){
            count = 0;
        }else {
            count += 1;
            if(count == DATASIZE) break;
        }

        m.backward(&m);
        m.updateParameter(&m);
        m.forward(&m);
        j++;
        //if(j == 200) break;
    }
*/
    printf("=============\n");

    for (int i=0; i<DATASIZE; i++){
        m.init(&m, a, &d[i]);
        m.forward(&m);
    }

    for (int j=0; j<1000; j++){
        printf("=============\n");
        for (int i=0; i<DATASIZE; i++){
            m.init(&m, a, &d[i]);
            m.forward(&m);
            m.backward(&m);
            m.updateParameter(&m);
            m.forward(&m);
        }
    }


    return 0;
}