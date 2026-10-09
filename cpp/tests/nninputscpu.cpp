// Standalone differential/performance harness. Not linked into the engine.
// Link twice with the original and candidate nninputs.cpp and compare dumps.
#include "../neuralnet/nninputs.h"
#include "../tests/tests.h"
#include <chrono>
#include <fstream>
#include <iostream>
#include <algorithm>

struct Position {
  Board board; BoardHistory hist; Player pla;
};
int main(int argc,char** argv) {
  if(argc!=2)return 2;
  Board::initHash(); ScoreValue::initTables();
  Tests::runNNInputsV3V4Tests();
  Tests::runExcludeTerritoryAtariNNInputsTests();
  Rand rand("round3-cpu-ladder");
  std::vector<Position> positions;
  for(int game=0;game<24;game++) {
    int x=game%3==0?9:19,y=game%4==0?13:x;
    Board board(x,y);Player pla=P_BLACK;
    Rules rules=Rules::parseRules(game%2==0?"tromp-taylor":"japanese");
    BoardHistory hist(board,pla,rules,0,BoardHistoryModes(game%2!=0,game%3!=0));
    for(int turn=0;turn<x*y*2 && !hist.isGameFinished;turn++) {
      if(turn%5==0)positions.push_back({board,hist,pla});
      std::vector<Loc> legal;
      for(int yy=0;yy<y;yy++)for(int xx=0;xx<x;xx++) {
        Loc loc=Location::getLoc(xx,yy,x);
        if(hist.isLegal(board,loc,pla))legal.push_back(loc);
      }
      Loc move=legal.empty()?Board::PASS_LOC:legal[rand.nextUInt((uint32_t)legal.size())];
      hist.makeBoardMoveAssumeLegal(board,move,pla,nullptr);pla=getOpp(pla);
    }
  }
  using Fill=void(*)(const Board&,const BoardHistory&,Player,const MiscNNInputParams&,int,int,bool,float*,float*);
  Fill fill[]={NNInputs::fillRowV3,NNInputs::fillRowV4,NNInputs::fillRowV5,NNInputs::fillRowV6,NNInputs::fillRowV7};
  const int spatial[]={22,22,13,22,22},global[]={14,14,12,16,19};
  std::vector<float> row(22*19*19),g(19);
  std::ofstream out(argv[1],std::ios::binary);
  size_t cases=0;
  for(const auto& p:positions)for(int v=0;v<5;v++)for(bool nhwc:{false,true})for(bool padded:{false,true})for(int history:{0,1000}){
    int x=padded?19:p.board.x_size,y=padded?19:p.board.y_size;
    MiscNNInputParams params;params.maxHistory=history;
    fill[v](p.board,p.hist,p.pla,params,x,y,nhwc,row.data(),g.data());
    out.write((char*)row.data(),spatial[v]*x*y*sizeof(float));out.write((char*)g.data(),global[v]*sizeof(float));cases++;
  }
  out.close();if(!out)return 3;
  std::cout<<"positions="<<positions.size()<<" cases="<<cases<<std::endl;
  // Same precomputed positions, no I/O or game generation in the timed region.
  volatile float sink=0;
  for(int rep=0;rep<5;rep++){
    auto start=std::chrono::steady_clock::now();
    for(int k=0;k<5;k++)for(const auto& p:positions){
      MiscNNInputParams params;
      NNInputs::fillRowV7(p.board,p.hist,p.pla,params,19,19,true,row.data(),g.data());sink+=row[0];
    }
    double ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
    std::cout<<"fillRowV7_ms="<<ms<<" calls="<<positions.size()*5<<std::endl;
  }
  ScoreValue::freeTables();return sink==123456789.0f?4:0;
}
