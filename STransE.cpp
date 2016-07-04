#include<iostream>
#include<cstring>
#include<cstdio>
#include<map>
#include<vector>
#include<string>
#include<ctime>
#include<cmath>
#include <algorithm>
#include<cstdlib>
#include<sstream>
#include <Eigen/Dense>
using namespace Eigen;
using namespace std;

#define pi 3.1415926535897932384626433832795

double rand(double min, double max) {
	return min + (max - min) * rand() / (RAND_MAX + 1.0);
}
double normal(double x, double miu, double sigma) {
	return 1.0 / sqrt(2 * pi) / sigma
			* exp(-1 * (x - miu) * (x - miu) / (2 * sigma * sigma));
}
double sigmod(double x) {
	return 1.0 / (1 + exp(-2 * x));
}
double randn(double miu, double sigma, double min, double max) {
	double x, y, dScope;
	do {
		x = rand(min, max);
		y = normal(x, miu, sigma);
		dScope = rand(0.0, normal(miu, miu, sigma));
	} while (dScope > y);
	return x;
}

double sqr(double x) {
	return x * x;
}

double vec_len(vector<double> &a) {
	double res = 0;
	for (int i = 0; i < a.size(); i++)
		res += a[i] * a[i];
	res = sqrt(res);
	return res;
}

string folder, suffix;
int nepoch = 1000, evalStep = 1000;
char buf[1000];
int relation_num, entity_num, nThreads = 4;
map<string, int> relation2id, entity2id;
map<int, string> id2entity, id2relation;

map<int, int> entity2num;

map<int, map<int, int> > left_entity, right_entity;
map<int, double> left_avg, right_avg;

int myrandom(int i) {
	return std::rand() % i;
}
double cmp(pair<int, double> a, pair<int, double> b) {
	return a.second < b.second;
}

class STransE {

public:
	map<pair<int, int>, map<int, int> > train_triples, dev_test_triples;
	map<int, double> headTailSelector;

	void add(int h, int r, int t) {
		kb_h.push_back(h);
		kb_r.push_back(r);
		kb_t.push_back(t);
		train_triples[make_pair(h, r)][t] = 1;
	}

	void addDev(int h, int r, int t) {
		dev_fb_h.push_back(h);
		dev_fb_r.push_back(r);
		dev_fb_t.push_back(t);
		dev_test_triples[make_pair(h, r)][t] = 1;
	}

	void addTest(int h, int r, int t) {
		test_fb_h.push_back(h);
		test_fb_r.push_back(r);
		test_fb_t.push_back(t);
		dev_test_triples[make_pair(h, r)][t] = 1;
	}

	void run(int n_in, double rate_in, double margin_in, bool l1_in,
			bool init_in, bool isSTransE_in) {
		n = n_in;
		rate = rate_in;
		margin = margin_in;
		L1_flag = l1_in;
		isInit = init_in;
		isSTransE = isSTransE_in;

		for (int i = 0; i < relation_num; i++) {
			headTailSelector[i] = 1000 * right_avg[i]
					/ (right_avg[i] + left_avg[i]);
		}

		relation_vec.resize(relation_num);
		for (int i = 0; i < relation_vec.size(); i++)
			relation_vec[i].resize(n);
		entity_vec.resize(entity_num);
		for (int i = 0; i < entity_vec.size(); i++)
			entity_vec[i].resize(n);

		if (isSTransE) {

			W1.resize(relation_num);
			for (int i = 0; i < relation_num; i++) {
				W1[i].resize(n, n);
				for (int jj = 0; jj < n; jj++) {
					for (int ii = 0; ii < n; ii++) {
						if (ii == jj)
							W1[i](jj, ii) = 1;
						else
							W1[i](jj, ii) = 0;
					}
				}
			}

			W2.resize(relation_num);
			for (int i = 0; i < relation_num; i++) {
				W2[i].resize(n, n);
				for (int jj = 0; jj < n; jj++) {
					for (int ii = 0; ii < n; ii++) {
						if (ii == jj)
							W2[i](jj, ii) = 1;
						else
							W2[i](jj, ii) = 0;
					}
				}
			}

			if (isInit) {
				FILE* f1 = fopen((folder + "entity2vec.init").c_str(), "r");
				for (int i = 0; i < entity_num; i++) {
					for (int ii = 0; ii < n; ii++)
						fscanf(f1, "%lf", &entity_vec[i][ii]);
					norm(entity_vec[i]);
				}
				fclose(f1);

				FILE* f2 = fopen((folder + "relation2vec.init").c_str(), "r");
				for (int i = 0; i < relation_num; i++) {
					for (int ii = 0; ii < n; ii++)
						fscanf(f2, "%lf", &relation_vec[i][ii]);
				}
				fclose(f2);

			} else {
				for (int i = 0; i < relation_num; i++) {
					for (int ii = 0; ii < n; ii++)
						relation_vec[i][ii] = randn(0, 1.0 / n, -6 / sqrt(n),
								6 / sqrt(n));
				}

				for (int i = 0; i < entity_num; i++) {
					for (int ii = 0; ii < n; ii++)
						entity_vec[i][ii] = randn(0, 1.0 / n, -6 / sqrt(n),
								6 / sqrt(n));
					norm(entity_vec[i]);
				}

				initialize();
			}

			optimize_STransE();

		} else {
			for (int i = 0; i < relation_num; i++) {
				for (int ii = 0; ii < n; ii++)
					relation_vec[i][ii] = randn(0, 1.0 / n, -6 / sqrt(n),
							6 / sqrt(n));
			}

			for (int i = 0; i < entity_num; i++) {
				for (int ii = 0; ii < n; ii++)
					entity_vec[i][ii] = randn(0, 1.0 / n, -6 / sqrt(n),
							6 / sqrt(n));
				norm(entity_vec[i]);
			}
			optimize_TransE();
		}
	}

private:
	int n;
	double cost, rate, margin;
	vector<int> kb_h, kb_t, kb_r;
	vector<int> dev_fb_h, dev_fb_t, dev_fb_r;
	vector<int> test_fb_h, test_fb_t, test_fb_r;
	vector<RowVectorXd> relation_vec, entity_vec, relation_tmp, entity_tmp;
	vector<MatrixXd> W1, W1_tmp, W2, W2_tmp;
	bool L1_flag = true, isInit = true, isSTransE = true;

	void norm(RowVectorXd &a) {
		if (a.norm() > 1)
			a.normalize();
	}

	void norm(RowVectorXd &a, MatrixXd &A) {
		while (true) {
			double x = (a * A).norm();
			if (x > 1) {
				for (int ii = 0; ii < n; ii++) {
					double tmp = A.col(ii).dot(a);
					for (int jj = 0; jj < n; jj++) {
						A(jj, ii) -= rate * tmp * a[jj];
						a[jj] -= rate * tmp * A(jj, ii);
					}
				}
			} else
				break;
		}
	}

	void optimize_STransE() {

		cout.precision(10);

		FILE* fLog = fopen((folder + "STransE" + suffix + ".log.txt").c_str(),
				"w");

		cout
				<< "Optimize entity vectors, relation vectors and relation matrices:"
				<< endl;
		fprintf(fLog, "%s\n",
				"Optimize entity vectors, relation vectors and relation matrices:");

		for (int epoch = 0; epoch < nepoch; epoch++) {

			cost = 0;

			relation_tmp = relation_vec;
			entity_tmp = entity_vec;
			W1_tmp = W1;
			W2_tmp = W2;

			for (int i = 0; i < kb_h.size(); ++i) {
				int sampledEn = rand() % entity_num;

				int head = kb_h[i], tail = kb_t[i], rel = kb_r[i];

				double pr = headTailSelector[rel];

				if (rand() % 1000 < pr) {

					while (train_triples[make_pair(head, rel)].count(sampledEn)
							> 0)
						sampledEn = rand() % entity_num;

					updateParas_STransE(head, rel, tail, head, rel, sampledEn);

					norm(relation_tmp[rel]);
					norm(entity_tmp[head]);
					norm(entity_tmp[tail]);
					norm(entity_tmp[sampledEn]);
					norm(entity_tmp[head], W1_tmp[rel]);
					norm(entity_tmp[tail], W2_tmp[rel]);
					norm(entity_tmp[sampledEn], W2_tmp[rel]);

				} else {
					while (train_triples[make_pair(sampledEn, rel)].count(tail)
							> 0)
						sampledEn = rand() % entity_num;

					updateParas_STransE(head, rel, tail, sampledEn, rel, tail);

					norm(relation_tmp[rel]);
					norm(entity_tmp[head]);
					norm(entity_tmp[tail]);
					norm(entity_tmp[sampledEn]);
					norm(entity_tmp[head], W1_tmp[rel]);
					norm(entity_tmp[tail], W2_tmp[rel]);
					norm(entity_tmp[sampledEn], W1_tmp[rel]);
				}

				relation_vec[rel] = relation_tmp[rel];
				W1[rel] = W1_tmp[rel];
				W2[rel] = W2_tmp[rel];
				entity_vec[head] = entity_tmp[head];
				entity_vec[tail] = entity_tmp[tail];
				entity_vec[sampledEn] = entity_tmp[sampledEn];

			}

			relation_vec = relation_tmp;
			entity_vec = entity_tmp;
			W1 = W1_tmp;
			W2 = W2_tmp;

			cout << "\tepoch " << epoch << " : " << cost << endl;
			fprintf(fLog, "\t%s %d : %.6lf\n", "---\nepoch ", epoch, cost);

			if ((epoch + 1) % evalStep == 0) {
				write(epoch);

				cout << "\tEvaluating on validation set: " << endl;
				double validScore = runEntityPrediction_Dev();
				cout << "\tFiltered MR on validation set: " << validScore
						<< endl;
				fprintf(fLog, "\t%s %.6lf\n", "Filtered MR on validation set:",
						validScore);

				cout << "\tEvaluating on test set: " << endl;
				vector<double> linkPredictionValues;
				runEntityPrediction_Test(linkPredictionValues);
				cout << "\tRaw scores (MR, MRR, H@1, H@5, H@10) on test set: "
						<< linkPredictionValues[0] << " "
						<< linkPredictionValues[1] << " "
						<< linkPredictionValues[2] << " "
						<< linkPredictionValues[3] << " "
						<< linkPredictionValues[4] << endl;
				fprintf(fLog, "\t%s %.6lf %.6lf %.6lf %.6lf %.6lf\n",
						"Raw scores (MR, MRR, H@1, H@5, H@10) on test set:",
						linkPredictionValues[0], linkPredictionValues[1],
						linkPredictionValues[2], linkPredictionValues[3],
						linkPredictionValues[4]);
				cout
						<< "\tFiltered scores (MR, MRR, H@1, H@5, H@10) on test set:"
						<< " " << linkPredictionValues[5] << " "
						<< linkPredictionValues[6] << " "
						<< linkPredictionValues[7] << " "
						<< linkPredictionValues[8] << " "
						<< linkPredictionValues[9] << endl;
				fprintf(fLog, "\t%s %.6lf %.6lf %.6lf %.6lf %.6lf\n",
						"Filtered scores (MR, MRR, H@1, H@5, H@10) on test set:",
						linkPredictionValues[5], linkPredictionValues[6],
						linkPredictionValues[7], linkPredictionValues[8],
						linkPredictionValues[9]);
			}
		}
		fclose(fLog);
	}

	void updateParas_STransE(int &e1_a, int &rel_a, int &e2_a, int &e1_b,
			int &rel_b, int &e2_b) {
		VectorXd temp1, temp2;
		double sum1 = getScore_STransE(e1_a, rel_a, e2_a, temp1);
		double sum2 = getScore_STransE(e1_b, rel_b, e2_b, temp2);
		if (sum1 + margin > sum2) {
			cost += margin + sum1 - sum2;
			SGDupdate_STransE(e1_a, rel_a, e2_a, temp1, 1);
			SGDupdate_STransE(e1_b, rel_b, e2_b, temp2, -1);
		}
	}

	double getScore_STransE(int &e1, int &rel, int &e2, VectorXd &d) {
		d = entity_vec[e1] * W1[rel] + relation_vec[rel]
				- entity_vec[e2] * W2[rel];
		if (L1_flag)
			return d.lpNorm<1>();
		else
			return d.squaredNorm();
	}

	double getScore_STransE(int &e1, int &rel, int &e2) {
		VectorXd d = entity_vec[e1] * W1[rel] + relation_vec[rel]
				- entity_vec[e2] * W2[rel];
		if (L1_flag)
			return d.lpNorm<1>();
		else
			return d.squaredNorm();
	}

	void SGDupdate_STransE(int &e1, int &rel, int &e2, VectorXd &d,
			int isCorrect) {
		for (int i = 0; i < n; i++) {
			double x = 2 * d[i];
			if (L1_flag)
				if (x > 0)
					x = 1;
				else
					x = -1;

			double tmp = isCorrect * rate * x;

			W1_tmp[rel].col(i) -= tmp * entity_vec[e1].transpose();
			W2_tmp[rel].col(i) += tmp * entity_vec[e2].transpose();
			entity_tmp[e1] -= tmp * W1[rel].col(i).transpose();
			entity_tmp[e2] += tmp * W2[rel].col(i).transpose();
			relation_tmp[rel][i] -= tmp;
		}
	}

	void initialize() {

		cout.precision(10);

		cout
				<< "STransE initialization: fix relation matrices as identity matrices"
				<< "\n\t and only optimize entity and relation vectors with 1000 epoches:"
				<< endl;

		for (int epoch = 0; epoch < 1000; epoch++) {

			cost = 0;

			relation_tmp = relation_vec;
			entity_tmp = entity_vec;

			for (int i = 0; i < kb_h.size(); ++i) {
				int sampledEn = rand() % entity_num;

				int head = kb_h[i], tail = kb_t[i], rel = kb_r[i];

				if (rand() % 1000 < 500) {

					while (train_triples[make_pair(head, rel)].count(sampledEn)
							> 0)
						sampledEn = rand() % entity_num;
					updateParas_TransE(head, rel, tail, head, rel, sampledEn);

				} else {
					while (train_triples[make_pair(sampledEn, rel)].count(tail)
							> 0)
						sampledEn = rand() % entity_num;
					updateParas_TransE(head, rel, tail, sampledEn, rel, tail);
				}

				norm(relation_tmp[rel]);
				norm(entity_tmp[head]);
				norm(entity_tmp[tail]);
				norm(entity_tmp[sampledEn]);

				relation_vec[rel] = relation_tmp[rel];
				entity_vec[head] = entity_tmp[head];
				entity_vec[tail] = entity_tmp[tail];
				entity_vec[sampledEn] = entity_tmp[sampledEn];

			}

			relation_vec = relation_tmp;
			entity_vec = entity_tmp;

			cout << "\tepoch " << epoch << " : " << cost << endl;
		}
	}

	void optimize_TransE() {

		cout.precision(10);

		FILE* fLog = fopen((folder + "TransE" + suffix + ".log.txt").c_str(),
				"w");

		cout << "Optimize entity and relation vectors:" << endl;
		fprintf(fLog, "%s\n", "Optimize entity and relation vectors:");

		for (int epoch = 0; epoch < nepoch; epoch++) {

			cost = 0;

			relation_tmp = relation_vec;
			entity_tmp = entity_vec;

			for (int i = 0; i < kb_h.size(); ++i) {
				int sampledEn = rand() % entity_num;

				int head = kb_h[i], tail = kb_t[i], rel = kb_r[i];

				double pr = headTailSelector[rel];
				if (rand() % 1000 < pr) {

					while (train_triples[make_pair(head, rel)].count(sampledEn)
							> 0)
						sampledEn = rand() % entity_num;
					updateParas_TransE(head, rel, tail, head, rel, sampledEn);

				} else {
					while (train_triples[make_pair(sampledEn, rel)].count(tail)
							> 0)
						sampledEn = rand() % entity_num;
					updateParas_TransE(head, rel, tail, sampledEn, rel, tail);
				}

				norm(relation_tmp[rel]);
				norm(entity_tmp[head]);
				norm(entity_tmp[tail]);
				norm(entity_tmp[sampledEn]);

				relation_vec[rel] = relation_tmp[rel];
				entity_vec[head] = entity_tmp[head];
				entity_vec[tail] = entity_tmp[tail];
				entity_vec[sampledEn] = entity_tmp[sampledEn];

			}

			relation_vec = relation_tmp;
			entity_vec = entity_tmp;

			cout << "\tepoch " << epoch << " : " << cost << endl;
			fprintf(fLog, "\t%s %d : %.6lf\n", "---\nepoch ", epoch, cost);

			if ((epoch + 1) % evalStep == 0) {
				write(epoch);

				cout << "\tEvaluating on validation set: " << endl;
				double validScore = runEntityPrediction_Dev();
				cout << "\tFiltered MR on validation set: " << validScore
						<< endl;
				fprintf(fLog, "\t%s %.6lf\n", "Filtered MR on validation set:",
						validScore);

				cout << "\tEvaluating on test set: " << endl;
				vector<double> linkPredictionValues;
				runEntityPrediction_Test(linkPredictionValues);
				cout << "\tRaw scores (MR, MRR, H@1, H@5, H@10) on test set: "
						<< linkPredictionValues[0] << " "
						<< linkPredictionValues[1] << " "
						<< linkPredictionValues[2] << " "
						<< linkPredictionValues[3] << " "
						<< linkPredictionValues[4] << endl;
				fprintf(fLog, "\t%s %.6lf %.6lf %.6lf %.6lf %.6lf\n",
						"Raw scores (MR, MRR, H@1, H@5, H@10) on test set:",
						linkPredictionValues[0], linkPredictionValues[1],
						linkPredictionValues[2], linkPredictionValues[3],
						linkPredictionValues[4]);
				cout
						<< "\tFiltered scores (MR, MRR, H@1, H@5, H@10) on test set:"
						<< " " << linkPredictionValues[5] << " "
						<< linkPredictionValues[6] << " "
						<< linkPredictionValues[7] << " "
						<< linkPredictionValues[8] << " "
						<< linkPredictionValues[9] << endl;
				fprintf(fLog, "\t%s %.6lf %.6lf %.6lf %.6lf %.6lf\n",
						"Filtered scores (MR, MRR, H@1, H@5, H@10) on test set:",
						linkPredictionValues[5], linkPredictionValues[6],
						linkPredictionValues[7], linkPredictionValues[8],
						linkPredictionValues[9]);
			}

		}

		fclose(fLog);
	}

	void updateParas_TransE(int &e1_a, int &rel_a, int &e2_a, int &e1_b,
			int &rel_b, int &e2_b) {
		VectorXd temp1, temp2;
		double sum1 = getScore_TransE(e1_a, rel_a, e2_a, temp1);
		double sum2 = getScore_TransE(e1_b, rel_b, e2_b, temp2);
		if (sum1 + margin > sum2) {
			cost += margin + sum1 - sum2;
			SGDupdate_TransE(e1_a, rel_a, e2_a, temp1, 1);
			SGDupdate_TransE(e1_b, rel_b, e2_b, temp2, -1);
		}
	}

	double getScore_TransE(int &e1, int &rel, int &e2, VectorXd &d) {
		d = entity_vec[e1] + relation_vec[rel] - entity_vec[e2];
		if (L1_flag)
			return d.lpNorm<1>();
		else
			return d.squaredNorm();
	}

	double getScore_TransE(int &e1, int &rel, int &e2) {
		VectorXd d = entity_vec[e1] + relation_vec[rel] - entity_vec[e2];
		if (L1_flag)
			return d.lpNorm<1>();
		else
			return d.squaredNorm();
	}

	void SGDupdate_TransE(int &e1, int &rel, int &e2, VectorXd &d,
			int isCorrect) {
		d = 2 * d;
		if (L1_flag) {
			for (int i = 0; i < n; i++) {
				if (d[i] > 0)
					d[i] = 1;
				else if (d[i] < 0)
					d[i] = -1;
			}
		}
		if (isSTransE)
			d = isCorrect * d * 0.001;
		else
			d = isCorrect * d * rate;

		relation_tmp[rel] -= d;
		entity_tmp[e1] -= d;
		entity_tmp[e2] += d;
	}

	void write(int epoch) {
		ostringstream ss;
		ss << (epoch + 1);
		if (isSTransE) {
			FILE* f1 = fopen(
					(folder + "STransE" + suffix + ".e" + ss.str()
							+ ".relation2vec").c_str(), "w");
			FILE* f2 = fopen(
					(folder + "STransE" + suffix + ".e" + ss.str()
							+ ".entity2vec").c_str(), "w");

			for (int i = 0; i < relation_num; i++) {
				for (int ii = 0; ii < n; ii++)
					fprintf(f1, "%.6lf\t", relation_vec[i][ii]);
				fprintf(f1, "\n");
			}
			for (int i = 0; i < entity_num; i++) {
				for (int ii = 0; ii < n; ii++)
					fprintf(f2, "%.6lf\t", entity_vec[i][ii]);
				fprintf(f2, "\n");
			}

			fclose(f1);
			fclose(f2);

			FILE* f3 =
					fopen(
							(folder + "STransE" + suffix + ".e" + ss.str()
									+ ".W1").c_str(), "w");
			FILE* f4 =
					fopen(
							(folder + "STransE" + suffix + ".e" + ss.str()
									+ ".W2").c_str(), "w");
			for (int i = 0; i < relation_num; i++)
				for (int jj = 0; jj < n; jj++) {
					for (int ii = 0; ii < n; ii++) {
						fprintf(f3, "%.6lf\t", W1[i](jj, ii));
					}
					fprintf(f3, "\n");
				}

			for (int i = 0; i < relation_num; i++)
				for (int jj = 0; jj < n; jj++) {
					for (int ii = 0; ii < n; ii++) {
						fprintf(f4, "%.6lf\t", W2[i](jj, ii));
					}
					fprintf(f4, "\n");
				}

			fclose(f3);
			fclose(f4);
		}

		else {
			FILE* f1 = fopen(
					(folder + "TransE" + suffix + ".e" + ss.str()
							+ ".relation2vec").c_str(), "w");
			FILE* f2 = fopen(
					(folder + "TransE" + suffix + ".e" + ss.str()
							+ ".entity2vec").c_str(), "w");

			for (int i = 0; i < relation_num; i++) {
				for (int ii = 0; ii < n; ii++)
					fprintf(f1, "%.6lf\t", relation_vec[i][ii]);
				fprintf(f1, "\n");
			}
			for (int i = 0; i < entity_num; i++) {
				for (int ii = 0; ii < n; ii++)
					fprintf(f2, "%.6lf\t", entity_vec[i][ii]);
				fprintf(f2, "\n");
			}
			fclose(f1);
			fclose(f2);
		}
	}

	double runEntityPrediction_Dev() {

		vector<double> values;
		for (int i = 0; i < 10; i++)
			values.push_back(0.0);

		int devSize = dev_fb_h.size();
		int bSize = devSize / nThreads;
		map<int, vector<double> > results;

#pragma omp parallel for num_threads(nThreads)
		for (int i = 0; i < nThreads; i++) {
			int start = i * bSize;
			int end = (i + 1) * bSize;
			results[i] = evalEntityPrediction(isSTransE, start, end,
					train_triples, dev_test_triples, dev_fb_h, dev_fb_t,
					dev_fb_r);
		}

		results[nThreads] = evalEntityPrediction(isSTransE, nThreads * bSize,
				devSize, train_triples, dev_test_triples, dev_fb_h, dev_fb_t,
				dev_fb_r);

		for (map<int, vector<double> >::iterator it = results.begin();
				it != results.end(); it++) {
			vector<double> temp = it->second;
			for (int i = 0; i < 10; i++)
				values[i] += temp[i];
		}

		return values[5];
	}

	void runEntityPrediction_Test(vector<double> &evalValues) {

		vector<double> values;
		for (int i = 0; i < 10; i++)
			values.push_back(0.0);

		int testSize = test_fb_h.size();
		int bSize = testSize / nThreads;
		map<int, vector<double> > results;

#pragma omp parallel for num_threads(nThreads)
		for (int i = 0; i < nThreads; i++) {
			int start = i * bSize;
			int end = (i + 1) * bSize;
			results[i] = evalEntityPrediction(isSTransE, start, end,
					train_triples, dev_test_triples, test_fb_h, test_fb_t,
					test_fb_r);
		}

		results[nThreads] = evalEntityPrediction(isSTransE, nThreads * bSize,
				testSize, train_triples, dev_test_triples, test_fb_h, test_fb_t,
				test_fb_r);

		for (map<int, vector<double> >::iterator it = results.begin();
				it != results.end(); it++) {
			vector<double> temp = it->second;
			for (int i = 0; i < 10; i++)
				values[i] += temp[i];
		}

		evalValues = values;
	}

	vector<double> evalEntityPrediction(bool isSTransE, int start, int end,
			map<pair<int, int>, map<int, int> > triples,
			map<pair<int, int>, map<int, int> > dev_test_triples,
			vector<int> test_fb_h, vector<int> test_fb_t,
			vector<int> test_fb_r) {

		vector<double> values;
		for (int i = 0; i < 10; i++)
			values.push_back(0.0);

		double headMR = 0, tailMR = 0, headH10 = 0, headH1 = 0, headH5 = 0,
				tailH10 = 0, headMRR = 0, tailMRR = 0, tailH1 = 0, tailH5 = 0;

		double filter_headMR = 0, filter_tailMR = 0, filter_headH10 = 0,

		filter_headH1 = 0, filter_headH5 = 0, filter_tailH10 = 0,
				filter_headMRR = 0, filter_tailMRR = 0, filter_tailH1 = 0,
				filter_tailH5 = 0;

		for (int validid = start; validid < end; validid++) {
			//printf(" %d \n", validid);

			int head = test_fb_h[validid];
			int tail = test_fb_t[validid];
			int rel = test_fb_r[validid];
			vector<pair<int, double> > scores;
			for (int i = 0; i < entity_num; i++) {
				double sim = 0;
				if (isSTransE)
					sim = getScore_STransE(i, rel, tail);
				else
					sim = getScore_TransE(i, rel, tail);
				scores.push_back(make_pair(i, sim));
			}
			sort(scores.begin(), scores.end(), cmp);

			int filter = 0;
			for (int i = 0; i < scores.size(); i++) {

				if ((triples[make_pair(scores[i].first, rel)].count(tail) == 0)
						&& (dev_test_triples[make_pair(scores[i].first, rel)].count(
								tail) == 0))
					filter += 1;

				if (scores[i].first == head) {

					headMR += (i + 1);
					headMRR += 1.0 / (i + 1);
					if (i == 0)
						headH1 += 1;
					if (i < 5)
						headH5 += 1;
					if (i < 10)
						headH10 += 1;

					filter_headMR += (filter + 1);
					filter_headMRR += 1.0 / (filter + 1);
					if (filter == 0)
						filter_headH1 += 1;
					if (filter < 5)
						filter_headH5 += 1;
					if (filter < 10)
						filter_headH10 += 1;

					break;
				}
			}
			scores.clear();

			for (int i = 0; i < entity_num; i++) {
				double sim = 0;
				if (isSTransE)
					sim = getScore_STransE(head, rel, i);
				else
					sim = getScore_TransE(head, rel, i);
				scores.push_back(make_pair(i, sim));
			}
			sort(scores.begin(), scores.end(), cmp);

			filter = 0;
			for (int i = 0; i < scores.size(); i++) {

				if ((triples[make_pair(head, rel)].count(scores[i].first) == 0)
						&& (dev_test_triples[make_pair(head, rel)].count(
								scores[i].first) == 0))
					filter += 1;

				if (scores[i].first == tail) {

					tailMR += (i + 1);
					tailMRR += 1.0 / (i + 1);

					if (i == 0)
						tailH1 += 1;
					if (i < 5)
						tailH5 += 1;
					if (i < 10)
						tailH10 += 1;

					filter_tailMR += (filter + 1);
					filter_tailMRR += 1.0 / (filter + 1);
					if (filter == 0)
						filter_tailH1 += 1;
					if (filter < 5)
						filter_tailH5 += 1;
					if (filter < 10)
						filter_tailH10 += 1;

					break;
				}
			}
		}

		double test_size = 1.0 * test_fb_h.size();

		values[0] = (headMR + tailMR) / (2 * test_size);
		values[1] = (headMRR + tailMRR) / (2 * test_size);
		values[2] = (headH1 + tailH1) / (2 * test_size);
		values[3] = (headH5 + tailH5) / (2 * test_size);
		values[4] = (headH10 + tailH10) / (2 * test_size);
		values[5] = (filter_headMR + filter_tailMR) / (2 * test_size);
		values[6] = (filter_headMRR + filter_tailMRR) / (2 * test_size);
		values[7] = (filter_headH1 + filter_tailH1) / (2 * test_size);
		values[8] = (filter_headH5 + filter_tailH5) / (2 * test_size);
		values[9] = (filter_headH10 + filter_tailH10) / (2 * test_size);

		return values;

	}
};

STransE stranse;
void readData() {
	FILE* f1 = fopen((folder + "entity2id.txt").c_str(), "r");
	FILE* f2 = fopen((folder + "relation2id.txt").c_str(), "r");
	int x;
	while (fscanf(f1, "%s%d", buf, &x) == 2) {
		string st = buf;
		entity2id[st] = x;
		id2entity[x] = st;
		entity_num++;
	}
	while (fscanf(f2, "%s%d", buf, &x) == 2) {
		string st = buf;
		relation2id[st] = x;
		id2relation[x] = st;
		relation_num++;
	}

	FILE* f_kb = fopen((folder + "train.txt").c_str(), "r");
	while (fscanf(f_kb, "%s", buf) == 1) {
		string head = buf; //left entity

		fscanf(f_kb, "%s", buf);
		string rel = buf;	//relation

		fscanf(f_kb, "%s", buf);
		string tail = buf;	//right entity

		if (entity2id.count(head) == 0) {
			cout << "miss entity:" << head << endl;
		}
		if (entity2id.count(tail) == 0) {
			cout << "miss entity:" << tail << endl;
		}
		if (relation2id.count(rel) == 0) {
			cout << "miss relation:" << rel << endl;
			relation2id[rel] = relation_num;
			relation_num++;
		}
		left_entity[relation2id[rel]][entity2id[head]]++;
		right_entity[relation2id[rel]][entity2id[tail]]++;

		//Input: left/head entity, right/tail entity, relation
		stranse.add(entity2id[head], relation2id[rel], entity2id[tail]);
	}
	for (int i = 0; i < relation_num; i++) {
		double sum1 = 0, sum2 = 0;
		for (map<int, int>::iterator it = left_entity[i].begin();
				it != left_entity[i].end(); it++) {
			sum1++;
			sum2 += it->second;
		}
		left_avg[i] = sum2 / sum1;
	}
	for (int i = 0; i < relation_num; i++) {
		double sum1 = 0, sum2 = 0;
		for (map<int, int>::iterator it = right_entity[i].begin();
				it != right_entity[i].end(); it++) {
			sum1++;
			sum2 += it->second;
		}
		right_avg[i] = sum2 / sum1;
	}
	cout << "#relations = " << relation_num << endl;
	cout << "#entities = " << entity_num << endl;
	fclose(f_kb);

	f_kb = fopen((folder + "valid.txt").c_str(), "r");
	while (fscanf(f_kb, "%s", buf) == 1) {
		string head = buf;

		fscanf(f_kb, "%s", buf);
		string rel = buf;

		fscanf(f_kb, "%s", buf);
		string tail = buf;

		if (entity2id.count(head) == 0) {
			cout << "miss entity:" << head << endl;
		}
		if (entity2id.count(tail) == 0) {
			cout << "miss entity:" << tail << endl;
		}
		if (relation2id.count(rel) == 0) {
			cout << "miss relation:" << rel << endl;
			relation2id[rel] = relation_num;
			relation_num++;
		}

		stranse.addDev(entity2id[head], relation2id[rel], entity2id[tail]);
	}
	fclose(f_kb);

	f_kb = fopen((folder + "test.txt").c_str(), "r");
	while (fscanf(f_kb, "%s", buf) == 1) {
		string head = buf;

		fscanf(f_kb, "%s", buf);
		string rel = buf;

		fscanf(f_kb, "%s", buf);
		string tail = buf;

		if (entity2id.count(head) == 0) {
			cout << "miss entity:" << head << endl;
		}
		if (entity2id.count(tail) == 0) {
			cout << "miss entity:" << tail << endl;
		}
		if (relation2id.count(rel) == 0) {
			cout << "miss relation:" << rel << endl;
			relation2id[rel] = relation_num;
			relation_num++;
		}
		stranse.addTest(entity2id[head], relation2id[rel], entity2id[tail]);
	}
	fclose(f_kb);
}

int ArgPos(char *str, int argc, char **argv) {
	int a;
	for (a = 1; a < argc; a++)
		if (!strcmp(str, argv[a])) {
			if (a == argc - 1) {
				printf("Argument missing for %s\n", str);
				exit(1);
			}
			return a;
		}
	return -1;
}

int main(int argc, char**argv) {
	srand((unsigned) time(NULL));

	int i, counter = 0;

	bool isSTransE = 1;
	if ((i = ArgPos((char *) "-model", argc, argv)) > 0) {
		isSTransE = atoi(argv[i + 1]);
		counter += 1;
	}

	if ((i = ArgPos((char *) "-data", argc, argv)) > 0) {
		folder = argv[i + 1];
		counter += 1;
	}

	int n = 50;
	if ((i = ArgPos((char *) "-size", argc, argv)) > 0) {
		n = atoi(argv[i + 1]);
		suffix += ".s" + string(argv[i + 1]);
		counter += 1;
	}

	double rate = 0.0001;
	if ((i = ArgPos((char *) "-lrate", argc, argv)) > 0) {
		rate = atof(argv[i + 1]);
		suffix += ".r" + string(argv[i + 1]);
		counter += 1;
	}

	double margin = 1.0;
	if ((i = ArgPos((char *) "-margin", argc, argv)) > 0) {
		margin = atof(argv[i + 1]);
		suffix += ".m" + string(argv[i + 1]);
		counter += 1;
	}

	bool l1 = 1;
	if ((i = ArgPos((char *) "-l1", argc, argv)) > 0) {
		l1 = atoi(argv[i + 1]);
		suffix += ".l1_" + string(argv[i + 1]);
		counter += 1;
	}

	bool ranInit = 1;
	if ((i = ArgPos((char *) "-init", argc, argv)) > 0) {
		ranInit = atoi(argv[i + 1]);
		suffix += ".i_" + string(argv[i + 1]);
		//counter += 1;
	}

	nepoch = 1000;
	if (isSTransE)
		nepoch = 2000;
	if ((i = ArgPos((char *) "-nepoch", argc, argv)) > 0) {
		nepoch = atoi(argv[i + 1]);
		//counter += 1;
	}

	evalStep = nepoch;
	if ((i = ArgPos((char *) "-evalStep", argc, argv)) > 0) {
		evalStep = atoi(argv[i + 1]);
		//counter += 1;
	}

	nThreads = 1;
	if ((i = ArgPos((char *) "-nthreads", argc, argv)) > 0) {
		nThreads = atoi(argv[i + 1]);
		//counter += 1;
	}

	if (counter < 6) {
		cout
				<< "Required hyper-parameters: -model, -data, -size, -margin, -l1, -lrate"
				<< endl;
		return 0;
	}

	string model = isSTransE == 1 ? "STransE" : "TransE";
	cout << "Model: " << model << endl;
	cout << "Dataset: " << folder << endl;
	cout << "Number of epoches: " << nepoch << endl;
	cout << "Vector size: " << n << endl;
	cout << "Margin: " << margin << endl;
	cout << "L1-norm: " << l1 << endl;
	cout << "SGD learing rate: " << rate << endl;
	//cout << "nThreads: " << nThreads << endl;

	readData();

	stranse.run(n, rate, margin, l1, ranInit, isSTransE);
}

