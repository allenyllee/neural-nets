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


    //Model1 m;
    //m.init = _init;


    int j=0;
    int i=0;
    srand(time(NULL));

    while (1){
        printf("loop %d\n",j);
        //i = rand() % DATASIZE ;
        i = j % DATASIZE ;

        Model1 m;
        m.init = _init;
        m.init(&m, a, &d[i]);
        m.forward(&m);

        if ( m.data->label * m.out->value < 0 ){
            count = 0;
        }else {
            count += 1;
            printf("count = %d\n",count);
            if(count == DATASIZE) break;
        }
/*
        if ( m.forward(&m) * m.data->label <0 ){
            count = 0;
        }else {
            count += 1;
            if(count == DATASIZE) break;
        }
*/
        //printf("debug4\n");
        m.backward(&m);
        //printf("debug5\n");
        m.updateParameter(&m);

        m.forward(&m);
        j++;
        //if(j == 10) break;
    }


    printf("=============\n");

    for (int i=0; i<DATASIZE; i++){
        Model1 m;
        m.init = _init;
        m.init(&m, a, &d[i]);
        m.forward(&m);
    }

/*
    for (int j=0; j<10000; j++){
        printf("=============\n");
        for (int i=0; i<DATASIZE; i++){
            //Model1 m;
            //m.init = _init;
            m.init(&m, a, &d[i]);
            //m.forward(&m);
            if (m.forward(&m) * m.data->label < 0) {
                //printf("forward value = %f, label = %d\n", m.out->value, m.data->label);
                m.backward(&m);
                m.updateParameter(&m);
            }
            //m.forward(&m);
        }
    }
*/

    return 0;
}