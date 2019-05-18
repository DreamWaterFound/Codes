#include <iostream>
#include <cmath>
#include <fstream>

#include <opencv2/core/core.hpp>

using namespace std;


void saveMat_mxn(ofstream& ofs,double *A,int m,int n);
void saveMat_mxn(ofstream& ofs,double **A,int m,int n);

//接下来把一般实矩阵的QR分解按函数的形式稍稍改写一下，输入是一般mxn实矩阵A，以及矩阵的行数m列数n，输出是QR形式的正交矩阵和上三角矩阵的乘积，

bool Maqr(double ** A, int m, int n)//进行一般实矩阵QR分解的函数
{
	ofstream ofs_eta("Way1_etas.txt");
	ofs_eta<<"Way 1 etas: \t We have "<<n<<" cols in the matrix."<<endl;
	ofs_eta<<"Origin Matrix: "<<endl;
		for (int i = 0; i<m; i++)
	{
		for (int j = 0; j<n; j++)
		{
			cout << "    " << A[i][j];
		}
		cout << endl;
	}
	saveMat_mxn(ofs_eta,A,m,n);
	ofs_eta<<endl;

	

	int i, j, k, nn, jj;
	double u, alpha, w, t;

    // 矩阵Q的大小是 m x m 的
	double** Q = new double*[m];   //动态分配内存空间
	for (i = 0; i<m; i++) Q[i] = new double[m];
    // ? 为啥? 不是所有的矩阵都可以进行QR分解吗?
	if (m < n)
	{
		cout << "\nQR分解失败！" << endl;
		return false;
	}

    // 给Q矩阵赋初始值
	for (i = 0; i <= m - 1; i++)
		for (j = 0; j <= m - 1; j++)
		{
			Q[i][j] = 0.0;
			if (i == j) Q[i][j] = 1.0;
		}

    // 确定列数
	nn = n;
	if (m == n) nn = m - 1;


    //在大循环k：0~m当中，进行H矩阵的求解，左乘Q，以及左乘A  k表示当前处理的列,也对应着处理的矩阵H
	for (k = 0; k <= nn - 1; k++)
    {
		ofs_eta<<endl<<endl<<"============== at col "<<k<<" =============="<<endl<<endl;

        // 储存这一列中的具有最大绝对值的那个元素的绝对值, u相当于书中的符号 \eta
		u = 0.0;
		for (i = k; i <= m - 1; i++)
		{
			w = fabs(A[i][k]);
			if (w > u) u = w;
		}

		ofs_eta<<"eta at col "<<k<<": \t\t"<<u<<endl;

        // 这个相当于 ORB EPnP中的 \sigma, 表示的是这一列的模长.下面的过程就是计算这个模长\alpha
		alpha = 0.0;
        // 遍历的是当前列的对角线下面的元素
		for (i = k; i <= m - 1; i++)
		{
			t = A[i][k] / u; alpha = alpha + t * t;
		}
		if (A[k][k] > 0.0) u = -u;
		alpha = u * sqrt(alpha);
		if (fabs(alpha) + 1.0 == 1.0)
		{
			cout << "\nQR分解失败！" << endl;
            return false;
		}

		ofs_eta<<"alpha at col "<<k<<": \t"<<alpha<<endl;

        // 原先的 \eta 用不到了,这里计算的的是 \rho
		u = sqrt(2.0*alpha*(alpha - A[k][k]));

		ofs_eta<<"rho at col "<<k<<": \t\t"<<u<<endl;

        //  \rho 不等于0才继续;如果等于0那么就完成这列的处理
		if ((u + 1.0) != 1.0)
		{

			ofs_eta<<"A without H at col "<<k<<":"<<endl;
			saveMat_mxn(ofs_eta,A,m,n);

            // 对应的A中的列元素中也用不到了,我们将计算得到的u_k直接存到原来的矩阵A中
			A[k][k] = (A[k][k] - alpha) / u;                                    //对角线元素
			for (i = k + 1; i <= m - 1; i++) A[i][k] = A[i][k] / u;             //对角线下面的元素

			ofs_eta<<"A with H at col "<<k<<":"<<endl;
			saveMat_mxn(ofs_eta,A,m,n);

			
			//以上就是H矩阵的求得，实际上程序并没有设置任何数据结构来存储H矩
			//阵，而是直接将u向量的元素赋值给原A矩阵的原列向量相应的位置，这样做
			//这样做是为了计算左乘矩阵Q和A


			for (j = 0; j <= m - 1; j++)
			{
				t = 0.0;
				for (jj = k; jj <= m - 1; jj++)
					t = t + A[jj][k] * Q[jj][j];
				for (i = k; i <= m - 1; i++)
					Q[i][j] = Q[i][j] - 2.0*t*A[i][k];
			}
            //左乘矩阵Q，循环结束后得到一个矩阵，再将这个矩阵转置一下就得到QR分解中的Q矩阵
            //也就是正交矩阵

			for (j = k + 1; j <= n - 1; j++)
			{
				t = 0.0;
				for (jj = k; jj <= m - 1; jj++)
					t = t + A[jj][k] * A[jj][j];
				for (i = k; i <= m - 1; i++)
					A[i][j] = A[i][j] - 2.0*t*A[i][k];
			}
			//H矩阵左乘A矩阵，循环完成之后，其上三角部分的数据就是上三角矩阵R
			A[k][k] = alpha;
			for (i = k + 1; i <= m - 1; i++)  A[i][k] = 0.0;
		}
	}

    
	for (i = 0; i <= m - 2; i++)
		for (j = i + 1; j <= m - 1; j++)
		{
			t = Q[i][j]; Q[i][j] = Q[j][i]; Q[j][i] = t;
		}
	//QR分解完毕，然后在函数体里面直接将Q、R矩阵打印出来
	for (i = 0; i<m; i++)
	{
		for (j = 0; j<n; j++)
		{
			cout << "\t" << Q[i][j];
		}
		cout << endl;
	}
	cout << endl;
	for (i = 0; i<m; i++)
	{
		for (j = 0; j<n; j++)
		{
			cout << "    " << A[i][j];
		}
		cout << endl;
	}


	ofs_eta.close();



    return true;
}


void qr_solve(double * A,int m,int n)
{
  static int max_nr = 0;        //? 静态的,存储运行这个程序历史上的最大的系数矩阵行数?
  static double * A1, * A2;     //? unkown

  const int nr = m;       // 系数矩阵A的行数
  const int nc = n;       // 系数矩阵A的列数

  // 判断是否需要重新分配A1 A2的内存区域
  if (max_nr != 0 && max_nr < nr) 
  {
    // 如果 max_nr != 0 说明之前已经创建了一个 last_max_nr < nr 的数组,不够我们现在使用了,需要重新分配内存;但是在重新分配之前我们需要先删除之前创建的内容
    delete [] A1;
    delete [] A2;
  }
  if (max_nr < nr) 
  {
    max_nr = nr;
    A1 = new double[nr];
    A2 = new double[nr];
  }

  double * pA = A,
         * ppAkk = pA;          // 一直都会指向对角线上的元素
  // 对系数矩阵的列展开遍历
  for(int k = 0; k < nc; k++) 
  {
    double * ppAik = ppAkk,           // 只是辅助下面的for循环中,遍历对角线元素下的当前列的所有元素
             eta = fabs(*ppAik);      // 存储当前列对角线元素下面的所有元素绝对值的最大值

    // 遍历当前对角线约束下,当前列的所有元素,并且找到它们中的最大的绝对值
    for(int i = k + 1; i < nr; i++) 
    {
      double elt = fabs(*ppAik);
      if (eta < elt) eta = elt;
        ppAik += nc;                  // 指向下一列
    }

    //? 判断靠谱不? 由于系数矩阵是雅克比,并且代价函数中的L元素都是二次项的形式,所以原则上都应该是大于0的
    if (eta == 0) 
    {
      A1[k] = A2[k] = 0.0;
      cerr << "God damnit, A is singular, this shouldn't happen." << endl;
      return;
    } 
    else
    {

      // 开始正儿八经地进行QR分解了
      // 感觉这里面使用的ription provided.是数值分析中的计算方法,和矩阵论中的定义的算法还是不一样的
      // 注意在这里面,ppAik被重ription provided.定义了,在这个结构中以这里定义的这个为准
      double * ppAik = ppAkk, 
              sum = 0.0,
              inv_eta = 1. / eta; // 卧槽还能直接+.表示浮点数啊,长见识了
      // 对当前列下面的每一行的元素展开遍历（包含位于矩阵主对角线上的元素）
      for(int i = k; i < nr; i++) 
      {
        *ppAik *= inv_eta;          // NOTICE 注意这个操作是永久的，当前指向的元素都会被“归一化”
        sum += *ppAik * *ppAik;     // 平方和
        ppAik += nc;                // 指针移动到下一行的这个元素
      }

      // 计算 sigma ,同时根据对角线元素的符号保持其为正数
      double sigma = sqrt(sum);
      if (*ppAkk < 0)               
        sigma = -sigma;
      
      *ppAkk += sigma;
      A1[k] = sigma * *ppAkk;
      A2[k] = -eta * sigma;
      // 对于后面的每一列展开遍历
      for(int j = k + 1; j < nc; j++) 
      {
        // 首先这一遍循环是为了计算tau
        // 又重新定义了
        double * ppAik = ppAkk, sum = 0;
        for(int i = k; i < nr; i++) 
        {
          sum += *ppAik * ppAik[j - k];
          ppAik += nc;
        }
        double tau = sum / A1[k];
        // 然后再一遍循环是为了修改
        ppAik = ppAkk;
        for(int i = k; i < nr; i++) 
        {
          ppAik[j - k] -= tau * *ppAik;
          ppAik += nc;
        }
      }
    }
    // 移动向下一个对角线元素
    ppAkk += nc + 1;
  }

  cout<<"OK:"<<endl;

	double t;
	pA=A;
	for (size_t i = 0; i <= m - 2; i++)
		for (size_t j = i + 1; j <= m - 1; j++)
		{
			t = pA[i*m+j]; pA[i*m+j] = pA[j*m+i]; pA[j*m+i] = t;
		}
	//QR分解完毕，然后在函数体里面直接将Q、R矩阵打印出来
	for (int i = 0; i<m; i++)
	{
		for (int j = 0; j<n; j++)
		{
			cout << "\t" << pA[i*m+j];
		}
		cout << endl;
	}
	cout << endl;



/*
  // b <- Qt b
  double * ppAjj = pA;// * pb = b->data.db;
  // 对于每一列展开计算
  for(int j = 0; j < nc; j++) 
  {
    // 这个部分倒的确是在计算Q^T*b
    double * ppAij = ppAjj, tau = 0;
    for(int i = j; i < nr; i++)	
    {
      tau += *ppAij * pb[i];
      ppAij += nc;
    }
    //? 但是后面我就看不懂了
    tau /= A1[j];
    ppAij = ppAjj;
    for(int i = j; i < nr; i++) 
    {
      pb[i] -= tau * *ppAij;`
      ppAij += nc;
    }
    ppAjj += nc + 1;
  }

  // X = R-1 b
  // backward method
  double * pX = X->data.db;
  pX[nc - 1] = pb[nc - 1] / A2[nc - 1];
  for(int i = nc - 2; i >= 0; i--) 
  {
    // 定位
    double * ppAij = pA + i * nc + (i + 1), sum = 0;

    for(int j = i + 1; j < nc; j++) 
    {
      sum += *ppAij * pX[j];    //pX[j] 就是上一步中刚刚计算出来的那个
      ppAij++;
    }
    pX[i] = (pb[i] - sum) / A2[i];  // 比较像了
  }
  */
}


int main(int argc, char* argv[])//主函数的调用
{
	int m, n;
	double** A1,** A2;
	cout<<"argc="<<argc<<endl;
	if(argc==2)
	{
		ifstream fs;
		// 文件格式: m n 数据
		fs.open(argv[1]);
		if(fs.fail())
		{
			cout<<"Error: Open file "<<argv[1]<<" failed."<<endl;
			return 0;
		}

		// 文件可以正常打开,但是为了又偷懒暂时不写文件读取错误的处理
		fs>>m>>n;
		cout<<"Matrix size: "<<m<<" x "<<n<<endl;

		A1 = new double* [m];
		A2 = new double* [m];
		for (int i = 0; i < m; i++) 
		{
			A1[i] = new double[n];
			A2[i] = new double[n];
		}

		for (int i = 0; i < m; i++)
		{
			for (int j = 0; j < n; j++)
			{
				fs >> A1[i][j];
				A2[i][j]=A1[i][j];
			}
		}

		cout<<"Load Complete."<<endl;

		fs.close();

	}
	else
	{
		cout << "输入矩阵的行数m，列数n" << endl;
		cin >> m >> n;
		A1 = new double* [m];
		A2 = new double* [m];
		for (int i = 0; i < m; i++) 
		{
			A1[i] = new double[n];
			A2[i] = new double[n];
		}
		cout << "输入矩阵A的每一个元素" << endl;
		for (int i = 0; i < m; i++)
		{
			for (int j = 0; j < n; j++)
			{
				cin >> A1[i][j];
				A2[i][j]=A1[i][j];
			}
		}
	}
	


	
	
			
	Maqr(A1, m, n);

	qr_solve(A2[0],m,n);

    // 这条不适用于Linux
	// system("pause");
	return 0;
}

void saveMat_mxn(ofstream& ofs,double *A,int m,int n)
{
	for (int i = 0; i<m; i++)
	{
		ofs<<"\t";
		for (int j = 0; j<n; j++)
		{
			ofs << "\t" << A[i*m+j];
		}
		ofs << endl;
	}
}

void saveMat_mxn(ofstream& ofs,double **A,int m,int n)
{
	for (int i = 0; i<m; i++)
	{
		ofs<<"\t";
		for (int j = 0; j<n; j++)
		{
			ofs << "\t" << A[i][j];
		}
		ofs << endl;
	}
}