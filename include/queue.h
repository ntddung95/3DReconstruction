#ifndef __QUEUE_H__
#define __QUEUE_H__

#include <iostream>
#include <opencv2/opencv.hpp>
//template<class Type>
class NODE{
  public:
    NODE *next;
    cv::Mat data;
    NODE(cv::Mat data_);
    ~NODE();
};

class QUEUE{
  public:
    int size;
    NODE *first;
    NODE *last;
    QUEUE();
    ~QUEUE();
    bool isEmpty();
    void Push(cv::Mat data);
    cv::Mat Pop();
};

#endif
