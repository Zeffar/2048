#include <iostream>
#include <vector>
#include <queue>
#include <algorithm>
#include <cmath>
using namespace std;
#define DEPTH 20
#define BEAM_WIDTH 200
#define pb push_back
#define NODE_COUNT 4205

long long seed;
uint8_t state[NODE_COUNT][4][4];
uint8_t root[NODE_COUNT];
int index{};
int depth{};
vector<pair<int, int>> OPEN;
queue<int> qe;

void move_up(int node, bool &ok);
void move_right(int node, bool &ok);
void move_down(int node, bool &ok);
void move_left(int node, bool &ok);

int evaluate (int node){ int score=0;
    vector<int> tiles; tiles.reserve(16);

    for(int i=0; i<4; i++)
        for(int j=0; j<4; j++) {
            int val=int(state[node][i][j]);
            if(val) tiles.pb(val);
        }
    sort(tiles.begin(), tiles.end(), greater<int>());
    //for(int it:tiles) cerr<<it<<' ';
    int cnt{};
    for(int i=3; i; i--)
        for(int j=3; j; j--)
        {
            if(!tiles[cnt]) return score;
            if(tiles[cnt++]==int(state[node][i][j])) score=score+int(state[node][i][j])*(i*4+j);
        }
    return score;
}


void spawn(int node){

    vector<int> freeCells; freeCells.reserve(16);
    for(int j=0; j<4; j++)
        for(int i=0; i<4; i++)
            if(int(state[node][i][j])==0) freeCells.pb(i+j*4);

    int spawnIndex = freeCells[(int)seed % freeCells.size()];
    int value = (seed & 0x10) == 0 ? 1 : 2;
    state[node][spawnIndex%4][spawnIndex/4] = value;
    //if(node<5) cerr<<"node "<<node<<": "<<spawnIndex<<'\n';
}


void extend (int node)
{

    bool ok=0;
    index++;
    if(!node) root[index]=index;
    else if(node<5) root[index]=root[node]*10+1;
    else root[index]=root[node];
    for(int i=0; i<4; i++)
        for(int j=0; j<4; j++)
            state[index][i][j]=state[node][i][j];
    move_up(index, ok);
    if(ok){
        spawn(index);
        OPEN.pb(make_pair(evaluate(index), index));
    }

    index++; ok=0;
    if(!node) root[index]=index;
    else if(node<5) root[index]=root[node]*10+2;
    else root[index]=root[node];
    for(int i=0; i<4; i++)
        for(int j=0; j<4; j++)
            state[index][i][j]=state[node][i][j];
    move_right(index, ok);
    if(ok){
            spawn(index);
    OPEN.pb(make_pair(evaluate(index), index));}

    index++; ok=0;
    if(!node) root[index]=index;
    else if(node<5) root[index]=root[node]*10+3;
    else root[index]=root[node];
    for(int i=0; i<4; i++)
        for(int j=0; j<4; j++)
            state[index][i][j]=state[node][i][j];
    move_down(index, ok);
    if(ok){spawn(index);
    OPEN.pb(make_pair(evaluate(index), index));}

    index++; ok=0;
    if(!node) root[index]=index;
    else if(node<5) root[index]=root[node]*10+4;
    else root[index]=root[node];
    for(int i=0; i<4; i++)
        for(int j=0; j<4; j++)
            state[index][i][j]=state[node][i][j];
    move_left(index, ok);
    if(ok){spawn(index);
    OPEN.pb(make_pair(evaluate(index), index));}
}


void outputMove(int node)
{
    int Move1=root[node]/10;
    int Move2=root[node]%10;
    switch (Move1){
    case 1: cout<<"U"; break;
    case 2: cout<<"R"; break;
    case 3: cout<<"D"; break;
    case 4: cout<<"L"; break;
    }
    switch (Move2){
    case 1: cout<<"U\n"; break;
    case 2: cout<<"R\n"; break;
    case 3: cout<<"D\n"; break;
    case 4: cout<<"L\n"; break;
    }
}

void outputState(int node){
    cerr<<"current state:\n";
    for(int i=0; i<4; i++)
    {
        for(int j=0; j<4; j++)
            cerr<<int(state[0][i][j])<<" ";
        cerr<<'\n';
    }
    cerr<<"next state:\n";
    for(int i=0; i<4; i++)
    {
        for(int j=0; j<4; j++)
            cerr<<int(state[int(root[node])][i][j])<<" ";
        cerr<<'\n';
    }


}

void clearState()
{
    index=0; depth=0;
    for(int it=0; it<NODE_COUNT; it++){ root[it]=0;
        for(int i=0; i<4; i++)
            for(int j=0; j<4; j++)
                state[it][i][j]=0;
    }
    OPEN.clear();
    while(!qe.empty()) qe.pop();
}


void find_move(){
    int NODE;
    int rez{};
    extend(0);
    seed=seed*seed%50515093L;
    while(depth<DEPTH){


        if(OPEN.size()<1) { break;}

        sort(OPEN.rbegin(), OPEN.rend());
        /*if(depth<2) {
            cerr<<NODE<<"\n";
            for(auto it:OPEN) cerr<<it.first<<" "<<it.second<<'\n';*/

        for(int i=0; i<BEAM_WIDTH && i<OPEN.size(); i++)
                qe.push(OPEN[i].second);
        OPEN.clear();
        depth++;
        rez=qe.front();
        while(!qe.empty())
        {
            NODE=qe.front();
            qe.pop();
            extend(NODE);
        }
        seed=seed*seed%50515093L;

    }
    outputMove(rez);
    //outputState(rez);
}


int main()
{
    while (1) {
        clearState();
        cin >> seed; cerr<<seed<<'\n';

        int score; cin >> score;

        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
                int x; cin>>x;
                if(x) state[0][i][j]=int(log2(x));
            }
        }
        //cout<<evaluate(0);
            find_move();
        }
    return 0;
}









//move functions
void move_left(int node, bool &ok){
    for(int i=0; i<4; i++){
        int n=0;
        int prev=0;
        for (int j=0; j<4; j++)
        {
            if (n==state[node][i][j] && n!=0){ ok=1;
                state[node][i][prev] = n+1;
                state[node][i][j] = 0;
                n = 0;
                continue;
            }
            if (state[node][i][j]!=0){
                n = state[node][i][j];
                prev = j;
            }
        }
    }
    for(int i=0; i<4; i++){
        for(int j=0; j<4; j++){
            for(int k=0; k<3; k++){
                if(state[node][i][k]==0 && state[node][i][k+1]!=0){ ok=1;
                    state[node][i][k]=state[node][i][k]^state[node][i][k+1];
                    state[node][i][k+1]=state[node][i][k]^state[node][i][k+1];
                    state[node][i][k]=state[node][i][k]^state[node][i][k+1];
                }
            }
        }
    }
}

void move_right(int node, bool &ok){
    for(int i=0; i<4; i++){
        int n=0;
        int prev=0;
        for (int j=3; j>=0; j--)
        {
            if (n==state[node][i][j] && n!=0){ ok=1;
                state[node][i][prev] = 1+n;
                state[node][i][j] = 0;
                n = 0;
                continue;
            }
            if (state[node][i][j]!=0){
                n = state[node][i][j];
                prev = j;
            }
        }
    }
    for(int i=0; i<4; i++){
        for(int j=0; j<4; j++){
            for(int k=3; k>0; k--){
                if(state[node][i][k]==0 && state[node][i][k-1]!=0){ ok=1;
                    state[node][i][k]=state[node][i][k]^state[node][i][k-1];
                    state[node][i][k-1]=state[node][i][k]^state[node][i][k-1];
                    state[node][i][k]=state[node][i][k]^state[node][i][k-1];
                }
            }
        }
    }

}

void move_up(int node, bool &ok){
    for(int i=0; i<4; i++){
        int n=0;
        int prev=0;
        for (int j=0; j<4; j++)
        {
            if (n==state[node][j][i] && n!=0){ ok=1;
                state[node][prev][i] = 1+n;
                state[node][j][i] = 0;
                n = 0;
                continue;
            }
            if (state[node][j][i]!=0){
                n = state[node][j][i];
                prev = j;
            }
        }
    }
    for(int i=0; i<4; i++){
        for(int j=0; j<4; j++){
            for(int k=0; k<3; k++){
                if(state[node][k][i]==0 && state[node][k+1][i]!=0){ ok=1;
                    state[node][k][i]=state[node][k][i]^state[node][k+1][i];
                    state[node][k+1][i]=state[node][k][i]^state[node][k+1][i];
                    state[node][k][i]=state[node][k][i]^state[node][k+1][i];
                }
            }
        }
    }

}

void move_down(int node, bool &ok){
    for(int i=0; i<4; i++){
        int n=0;
        int prev=0;
        for (int j=3; j>=0; j--)
        {
            if (n==state[node][j][i] && n!=0){ ok=1;
                state[node][prev][i] = 1+n;
                state[node][j][i] = 0;
                n = 0;
                continue;
            }
            if (state[node][j][i]!=0){
                n = state[node][j][i];
                prev = j;
            }
        }
    }
    for(int i=0; i<4; i++){
        for(int j=0; j<4; j++){
            for(int k=3; k>0; k--){
                if(state[node][k][i]==0 && state[node][k-1][i]!=0){ ok=1;
                    state[node][k][i]=state[node][k][i]^state[node][k-1][i];
                    state[node][k-1][i]=state[node][k][i]^state[node][k-1][i];
                    state[node][k][i]=state[node][k][i]^state[node][k-1][i];
                }
            }
        }
    }

}
