class transform():
    def __init__(self):
        self.gap= 5
        self.a_start_point = 5
        self.b_start_point = 5

    def to1dim(self,action):
        temp = (action[0]-self.a_start_point)//self.gap*90
        temp2 = (action[1]-self.a_start_point)//self.gap
        return int(temp+temp2)
    def to2dim(self,action):
        a = action//90
        a = 5+a*self.gap
        b = action%90
        b = 5+b*self.gap
        return [int(a),int(b)]
