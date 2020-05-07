class transform():
    def __init__(self):
        self.gap= 1
        self.a_start_point = 1
        self.b_start_point = 1

    def to1dim(self,action):
        temp = (action[0]-self.a_start_point)//self.gap*300
        temp2 = (action[1]-self.a_start_point)//self.gap
        return int(temp+temp2)
    def to2dim(self,action):
        a = action//300
        a = 1+a*self.gap
        b = action%300
        b = 1+b*self.gap
        return [int(a),int(b)]
