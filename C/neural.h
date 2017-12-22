#ifndef __NEURAL_H__
#define __NEURAL_H__

typedef struct _Unit Unit;
typedef struct _Neural Neural;
typedef struct _NeuralSet NeuralSet;
typedef Unit* (*ForwardFunc) (Neural*, Unit*, Unit*);
typedef void (*BackwardFunc) (Neural*);

struct _Unit{
    double value;
    double grad;
};

struct _Neural {
    Unit* u1;
    Unit* u2;
    Unit uout;
};

struct _NeuralSet {
    Neural* neurals;
    ForwardFunc forward;
    BackwardFunc backward;
};

/**
 * add gate
 */
Unit* add(Neural* const this, Unit* a, Unit* b){
    this->u1 = a;
    this->u2 = b;
    this->uout.value = this->u1->value + this->u2->value;
    return &this->uout;
}

void addgrad(Neural* const this){
    this->u1->grad += 1.0 * this->uout.grad;
    this->u2->grad += 1.0 * this->uout.grad;
}

void init_add(NeuralSet* const x, Neural* n){
    x->neurals = n;
    x->forward = add;
    x->backward = addgrad;
}

/**
 * multiply gate
 */
Unit* mul(Neural* const this, Unit* a, Unit* b){
    this->u1 = a;
    this->u2 = b;
    this->uout.value = this->u1->value * this->u2->value;
    return &this->uout;
}

void mulgrad(Neural* const this){
    this->u1->grad += this->u2->value * this->uout.grad;
    this->u2->grad += this->u1->value * this->uout.grad;
}

void init_mul(NeuralSet* const x, Neural* n){
    x->neurals = n;
    x->forward = mul;
    x->backward = mulgrad;
}
#endif