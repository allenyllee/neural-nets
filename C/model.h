#ifndef __MODEL_H__
#define __MODEL_H__

#include <stdio.h>
#include "./neural.h"
#include "./data.h"


#define PAR_NUM 3

/**
 * model 1:
 * f(x,y)=ax+by+c
 */
float model1(Unit* const parameter, Data const * const data, int isbackward){
    // function ax+by+c
    NeuralSet addset, mulset;
    Neural addneu[10], mulneu[10];
    Unit u[DIM];
    Unit *tmp1 = NULL, *tmp2 = NULL;
    Unit tmp3;
    float step_size = 0.001;

    for (int i=0; i<DIM; i++){
        u[i].value = data->x[i];
    }

    init_add(&addset, addneu);
    init_mul(&mulset, mulneu);

    // evaluate value
    tmp1 = mulset.forward(&mulset.neurals[0], &parameter[0], &u[0]); //ax
    tmp2 = mulset.forward(&mulset.neurals[1], &parameter[1], &u[1]); //by
    tmp1 = addset.forward(&addset.neurals[0], tmp1, tmp2); // ax+by
    tmp1 = addset.forward(&addset.neurals[1], tmp1, &parameter[2]); // ax+by+c

    printf("forward value = %f, label = %d\n", tmp1->value, data->label);

    tmp3.value = tmp1->value;

    if (isbackward){

        if (tmp1->value * data->label < 0) {
            // evaluate grad
            addset.neurals[1].uout.grad = 1;
            addset.backward(&addset.neurals[1]);
            addset.backward(&addset.neurals[0]);
            mulset.backward(&mulset.neurals[1]);
            mulset.backward(&mulset.neurals[0]);

            // update input varible
            step_size*=data->label;
            for (int i=0; i<PAR_NUM; i++){
                parameter[i].value += step_size * parameter[i].grad;
                parameter[i].grad = 0;
            }

            // evaluate value
            tmp1 = mulset.forward(&mulset.neurals[0], &parameter[0], &u[0]); //ax
            tmp2 = mulset.forward(&mulset.neurals[1], &parameter[1], &u[1]); //by
            tmp1 = addset.forward(&addset.neurals[0], tmp1, tmp2); // ax+by
            tmp1 = addset.forward(&addset.neurals[1], tmp1, &parameter[2]); // ax+by+c

            printf("forward value = %f\n", tmp1->value);
        }

    }

    return tmp3.value;
}


#endif