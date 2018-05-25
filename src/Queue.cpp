#include <iostream>
#include "queue.h"
#include <opencv2/opencv.hpp>

NODE::NODE(cv::Mat data_){
  cv::Mat input = data_.clone();
  this->data = input;
  this->next = NULL;
}

NODE::~NODE(){}

QUEUE::QUEUE(){
  this->size = 0;
  this->first = NULL;
  this->last = NULL;
}

QUEUE::~QUEUE(){}

bool QUEUE::isEmpty(){
  return (this->size == 0)?true:false;
}

void QUEUE::Push(cv::Mat data){
  NODE *tmp = new NODE(data);
  if (this->isEmpty()){
    this-> first = tmp;
    this-> last = tmp;
  }
  else{
    NODE *p = this->last;
    p->next = tmp;
    this->last = tmp;
  }
  this->size++;
}

cv::Mat QUEUE::Pop(){
  if (this->isEmpty()){
    std::cout<<"Queue is empty!"<<std::endl;
    return cv::Mat::zeros(3,3,0);
  }
  if (this->size == 1){
    this->size = 0;
    cv::Mat res = this->first->data;
    this->first = NULL;
    this->last = NULL;
    return res;
  }
  NODE *tmp = this->first;
  this->first = tmp->next;
  cv::Mat res = tmp->data;
  delete tmp;
  this->size--;
  return res;
}
