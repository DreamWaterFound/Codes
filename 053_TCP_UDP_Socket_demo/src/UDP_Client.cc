#include <stdio.h>   
#include <string.h>   
#include <errno.h>   
#include <stdlib.h>   
#include <unistd.h>   
#include <sys/types.h>   
#include <sys/socket.h>   
#include <netinet/in.h>   
#include <arpa/inet.h>   
   
  
#define DEST_PORT 8000   
#define DSET_IP_ADDRESS  "127.0.0.1"   
   
  
int main()  
{  
  /* socket文件描述符 */  
  int sock_fd;  
  
  /* 建立udp socket */  
  sock_fd = socket(AF_INET, SOCK_DGRAM, 0);  
  if(sock_fd < 0)  
  {  
    perror("socket");  
    exit(1);  
  }  
    
  /* 设置address */  
  struct sockaddr_in addr_serv;  
  int len;  
  memset(&addr_serv, 0, sizeof(addr_serv));  
  addr_serv.sin_family = AF_INET;  
  addr_serv.sin_addr.s_addr = inet_addr(DSET_IP_ADDRESS);  
  addr_serv.sin_port = htons(DEST_PORT);  
  len = sizeof(addr_serv);  
  
    
  int send_num;  
  int recv_num;  
  char send_buf[20] = "hey, who are you?";  
  char recv_buf[20];  
      
  printf("client send: %s\n", send_buf);  
    
  send_num = sendto(sock_fd, send_buf, strlen(send_buf), 0, (struct sockaddr *)&addr_serv, len);  
    
  if(send_num < 0)  
  {  
    perror("sendto error:");  
    exit(1);  
  }  
    
  recv_num = recvfrom(sock_fd, recv_buf, sizeof(recv_buf), 0, (struct sockaddr *)&addr_serv, (socklen_t *)&len);  
    
  if(recv_num < 0)  
  {  
    perror("recvfrom error:");  
    exit(1);  
  }  
    
  recv_buf[recv_num] = '\0';  
  printf("client receive %d bytes: %s\n", recv_num, recv_buf);  
    
  close(sock_fd);  
    
  return 0;  
}
