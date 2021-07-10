class EdgeContainer2 : IEdgeContainer
    {
        private readonly Dictionary<int, HashSet<int>> _comparisonLineIndicesArray = new Dictionary<int, HashSet<int>>();

        /// <summary>
        /// Adds a edge to this container.
        /// </summary>
        /// <param name="edge">Edge to add.</param>
        /// <returns>true if edge id added to the container. false otherwise.</returns>
        public bool AddEdge(Edge edge)
        {
            var index0 = edge.Index0;
            var index1 = edge.Index1;

            var found = Found(index0, index1) || Found(index1, index0);
            if (found)
            {
                return false;
            }

            // The edge is not found, we add it.
            HashSet<int> set;
            _comparisonLineIndicesArray.TryGetValue(index0, out set);

            if (set == null)
            {
                // The set is not created yet, we create it.
                set = new HashSet<int>();
                _comparisonLineIndicesArray.Add(index0, set);
            }

            // Add the edge.
            set.Add(index1);

            return true;
        }

        /// <summary>
        /// Return true if the oriented edge "index0-index1" is already in this container.
        /// </summary>
        private bool Found(int index0, int index1)
        {
            HashSet<int> set;
            _comparisonLineIndicesArray.TryGetValue(index0, out set);
            return (set != null) && set.Contains(index1);
        }

        public int Count
        {
            get { return _comparisonLineIndicesArray.Values.Sum(set => set.Count); }
        }
    }