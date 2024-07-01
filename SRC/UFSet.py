class DisjointSet:
    rank=[]
    def __init__(self, size):
        self.parent = [i for i in range(size)]
        self.rank = [0] * size

    def get_rank(self):
        return self.rank;

    def get_rep(self):
        rep=[]
        for i in self.parent:
            if i == self.parent[i] and self.rank[i] > 1:
                rep.append(i)
        return rep
    
    def find(self,i):
        if self.parent[i] == i:             
            return i
        else:
            rep = self.find(self.parent[i])
                     
            self.parent[i] = rep
            
            return rep
    

    def union(self, i, j):
        irep = self.find(i)
        jrep = self.find(j)
 
        if irep == jrep:
            return
 
        irank = self.rank[irep]
        jrank = self.rank[jrep]
 
        if irank < jrank: 
            self.parent[irep] = jrep
        elif jrank < irank:
            self.parent[jrep] = irep
        else:
            self.parent[irep] = jrep
            self.rank[jrep] += 1

 
