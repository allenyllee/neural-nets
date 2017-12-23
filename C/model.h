#ifndef __MODEL_H__
#define __MODEL_H__

#include <stdio.h>
#include "./neural.h"
#include "./data.h"


#define PAR_NUM 3
#define NEU_SIZE 10

typedef struct _model1 Model1;
typedef void (*modelInit) (Model1* this, Unit* const p, Data const * const d);
typedef float (*modelForward) (Model1* this);
typedef void (*modelBackward) (Model1* this);
typedef void (*modelUpdateParameter) (Model1* this);


void _init(Model1* this, Unit* const p, Data const * const d);
float _forward(Model1* this);
void _backward(Model1* this);
void _updateParameter(Model1* this);


struct _model1 {
    Unit *parameter;
    Data const *data;
    Unit *out, in[DIM];
    NeuralSet addset, mulset;
    Neural addneu[NEU_SIZE], mulneu[NEU_SIZE];
    modelInit init;
    modelForward forward;
    modelBackward backward;
    modelUpdateParameter updateParameter;
};

void _init(Model1* this, Unit* const p, Data const * const d){
    init_add(&this->addset, this->addneu);
    init_mul(&this->mulset, this->mulneu);

    // initialized neurals grad, to avoid strange thing when evaluate grad in backward function
    for(int i=0; i<NEU_SIZE; i++){
        this->addneu[i].uout.value = 0;
        this->addneu[i].uout.grad = 0;
        this->mulneu[i].uout.value = 0;
        this->mulneu[i].uout.grad = 0;
    }

    this->forward = _forward;
    this->backward = _backward;
    this->updateParameter = _updateParameter;
    this->parameter = p;
    this->data = d;

    // initializ input value to avoid backward probelm
    // in backward function, the last line is to process the very first varible passed in forward function
    // you should not use local varible to save data value in forward function
    // instead, use global varible declared in the structure
    for (int i=0; i<DIM; i++){
        this->in[i].value = this->data->x[i];
    }
}

float _forward(Model1* this){
    Unit *tmp1 = NULL;

    // evaluate value
    this->out = this->mulset.forward(&this->mulset.neurals[0], &this->parameter[0], &this->in[0]); //ax
    tmp1 = this->mulset.forward(&this->mulset.neurals[1], &this->parameter[1], &this->in[1]); //by
    this->out = this->addset.forward(&this->addset.neurals[0], this->out, tmp1); // ax+by
    this->out = this->addset.forward(&this->addset.neurals[1], this->out, &this->parameter[2]); // ax+by+c

    printf("forward value = %f, label = %d\n", this->out->value, this->data->label);

    return this->out->value;
}

void _backward(Model1* this){
    int pull=1;
/*
    pull=0;
    if ( this->data->label == 1 && this->out->value < 1){
        pull=1.0;
    }

    if ( this->data->label == -1 && this->out->value > -1){
        pull=-1.0;
    }
*/

    if (this->data->label * this->out->value < 0 ){
        pull *= this->data->label;
    }else {
        pull = 0;
    }

    // evaluate grad
    this->addset.neurals[1].uout.grad = pull;
    this->addset.backward(&this->addset.neurals[1]);
    this->addset.backward(&this->addset.neurals[0]);
    this->mulset.backward(&this->mulset.neurals[1]);
    this->mulset.backward(&this->mulset.neurals[0]);
}

void _updateParameter(Model1* this){
    float step_size = 0.001;
/*
    if (this->data->label * this->out->value < 0 ){
        step_size *= this->data->label;
    }else {
        step_size = 0;
    }
*/
    //printf("grad = ");
    // update input varible
    for (int i=0; i<PAR_NUM; i++){
        //printf("%f, ", this->parameter[i].grad);
        //this->parameter[i].grad += -this->parameter[i].value; //svm
        this->parameter[i].value += step_size * this->parameter[i].grad;
        this->parameter[i].grad = 0;
    }

    //printf("\n");
}


/**
 * model 1:
 * f(x,y)=ax+by+c
 */
float model1(Unit* const parameter, Data const * const data, int isbackward){
    Unit tmp3;
    Model1 m;

    m.init = _init;
    m.init(&m, parameter, data);

    tmp3.value = m.forward(&m);

    if (isbackward){
        m.backward(&m);
        m.updateParameter(&m);
        // evaluate value
        m.forward(&m);
    }

    return tmp3.value;
}





#endif