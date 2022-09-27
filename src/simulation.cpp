#include <vector>
#include <random>

template <typename NodeType>
class TreeHypergraph {
private:
	class Node {
	public:
		NodeType type;
		std::vector<std::vector<Node*>> edges;
	};

	std::vector<Node*> nodes;

public:
	unsigned n, r, k;
	double c;

	TreeHypergraph(unsigned n, unsigned r, double c, unsigned k) : 
		n(n), r(r), c(c), k(k) { }

	void tree_process_hypergraph() {
		std::random_device rd;
		std::mt19937 gen(rd());

		std::poisson_distribution<> dist((int) r * c);
	}
};
