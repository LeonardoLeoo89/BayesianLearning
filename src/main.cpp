#include <iostream>
#include <agrum/base.h>
 
int main() {
  gum::HashTable<std::string,int> h;
 
  h.insert("Hello",1);
  h.insert("World",2);
 
  std::cout << h << std::endl;
  return 0;
}
