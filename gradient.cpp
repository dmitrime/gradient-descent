#include <iostream>
#include <fstream>
#include <vector>
#include "gradient.h"

using namespace std;

ostream& operator<<(ostream& os, TrainingExample& te)
{
    int n = te.getFeatures().size();
    os << "(";
    for (int i = 0; i < n; i++)
        os << te.getFeature(i) << (i != n-1 ? ", " : ""); 
    os << ")";
    return os;
}

int main()
{
    int M = 0, N = 0;
    vector<TrainingExample> ts;
    ifstream f;
    f.open("data.txt");
    if (!f.is_open())
    {
        cout << "File not opened" << endl;
        return 1;
    }
    f >> M >> N;
    cout << "M = " << M << ", N = " << N << endl;
    for (int i = 0; i < M; i++)
    {
        vector<int> feat(N+1, 1);
        int y = 0;
        for (int j = 0; j < N; j++)
            f >> feat[j+1];
        f >> y;

        TrainingExample te(feat, y);
        ts.push_back(te);
    }
    f.close();

    for (unsigned i = 0; i < ts.size(); i++)
        cout << "Example " << i << ": " << ts[i] << endl;

    Hypothesis hyp(ts);
    vector<double> theta = hyp.gradientDescent();

    cout << endl;
    for (size_t i = 0; i < theta.size(); i++)
        cout << "th" << i << " = " << theta[i] << " ";
    cout << endl;

    cout << "Input x1 x2" << endl;
    int x1, x2;
    cin >> x1 >> x2;
    cout << "H = " << (theta[0]+theta[1]*x1+theta[2]*x2) << endl;
    

    return 0;
}

