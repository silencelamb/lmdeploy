# pipe_control-r0-source.cc

```cpp
// Same-host IPC control without Ray, Python, NumPy, Torch or accelerator runtime.
#include <array>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <sys/wait.h>
#include <unistd.h>

int64_t now() { timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec*1000000000LL+t.tv_nsec; }
void readall(int fd, void* ptr,size_t n) { auto p=(char*)ptr; while(n){ssize_t k=read(fd,p,n); if(k<=0)_exit(3);p+=k;n-=k;} }
void writeall(int fd,const void* ptr,size_t n) { auto p=(const char*)ptr; while(n){ssize_t k=write(fd,p,n);if(k<=0)_exit(4);p+=k;n-=k;} }
struct Reply{int64_t item,entered,exited;};
int main(int argc,char**argv){
 if(argc<4)return 2;
 int n=std::stoi(argv[2]),priority=std::stoi(argv[3]);
 std::ofstream out(argv[1]);
 std::array<int,2> tx,rx;std::array<pid_t,2> pids;
 for(int rank=0;rank<2;++rank){
  int a[2],b[2];pipe(a);pipe(b);pids[rank]=fork();
  if(!pids[rank]){
   close(a[1]);close(b[0]);if(priority)nice(priority);
   std::array<char,8192> payload;
   for(int i=0;i<n;++i){readall(a[0],payload.data(),payload.size());Reply r{i,now(),now()};writeall(b[1],&r,sizeof(r));}
   _exit(0);
  }
  close(a[0]);close(b[1]);tx[rank]=a[1];rx[rank]=b[0];
 }
 std::array<char,8192> payload{};
 out<<"{\"nice\":"<<priority<<",\"pids\":["<<pids[0]<<","<<pids[1]<<"],\"records\":[\n";
 double maxwait=0,maxdelivery=0;int slow=0;
 for(int i=0;i<n;++i){
  auto start=now();for(int rank=0;rank<2;++rank)writeall(tx[rank],payload.data(),payload.size());
  auto sent=now();timespec delay{0,12000000};nanosleep(&delay,nullptr);
  auto join=now();std::array<Reply,2> r;for(int rank=0;rank<2;++rank)readall(rx[rank],&r[rank],sizeof(Reply));auto end=now();
  double wait=(end-join)/1e6,delivery=(std::max(r[0].entered,r[1].entered)-start)/1e6;
  maxwait=std::max(maxwait,wait);maxdelivery=std::max(maxdelivery,delivery);slow+=wait>50||delivery>50;
  if(i)out<<",\n";
  out<<"{\"item\":"<<i<<",\"dispatch_ns\":"<<start<<",\"sent_ns\":"<<sent<<",\"join_ns\":"<<join<<",\"done_ns\":"<<end<<",\"entered_ns\":["<<r[0].entered<<","<<r[1].entered<<"]}";
 }
 out<<"\n],\"complete\":true}\n";
 for(auto pid:pids){int status;waitpid(pid,&status,0);if(status)return 5;}
 std::cout<<"max_wait_ms "<<maxwait<<" max_delivery_ms "<<maxdelivery<<" slow "<<slow<<std::endl;
}
```
