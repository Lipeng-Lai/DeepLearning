import sys

# 实现在迷宫A ~ B中找到一条最短路径
class Node():
    def __init__(self, state, parent, action):
        self.state = state # 表示该点的状态: " ", 还是"墙壁"
        self.parent = parent # 记录由 哪一个节点转移的：因为要记录路径
        self.action = action # 表示该点的 向下一个的点的移动方向：上下左右

# 用栈进行已经搜索过的节点: dfs
class StackFrontier():
    def __init__(self):
        self.frontier = [] # 初始化为空

    def add(self, node):
        self.frontier.append(node) # 搜索这个节点并把这个节点入栈

    # 问题: 这个函数的意思是什么 ?
    def contains_state(self, state):
        return any(node.state == state for node in self.frontier)

    def empty(self):
        return len(self.frontier) == 0 # 判空：栈中元素的个数为0

    def remove(self): # 弹出栈顶元素(数组模拟栈)
        if self.empty():
            raise Exception("empty frontier")
        else:
            node = self.frontier[-1] # 最后一个元素：(栈的特性先进后出)
            self.frontier = self.frontier[:-1]
            return node

# 问题：类里面的括号是什么意思 ?
# 用队列进行存储已经搜索过的节点: bfs
class QueueFrontier(StackFrontier):

    def remove(self):
        if self.empty():
            raise Exception("empty frontier")
        else:
            node = self.frontier[0] # 队列的特性：先进先出
            self.frontier = self.frontier[1:]
            return node

class Maze():

    def __init__(self, filename):

        # Read file and set height and width of maze
        with open(filename) as f:
            contents = f.read()

        # 验证 起点和终点
        if contents.count("A") != 1:
            raise Exception("maze must have exactly one start point")
        if contents.count("B") != 1:
            raise Exception("maze must have exactly one goal")

        # 根据文件匹配 高度和宽度
        contents = contents.splitlines()
        self.height = len(contents)
        self.width = max(len(line) for line in contents)

        # 预处理地图
        self.walls = []
        for i in range(self.height):
            row = []
            for j in range(self.width):
                try:
                    if contents[i][j] == "A":
                        self.start = (i, j)
                        row.append(False)
                    elif contents[i][j] == "B":
                        self.goal = (i, j)
                        row.append(False)
                    elif contents[i][j] == " ":
                        row.append(False)
                    else:
                        row.append(True) # 墙壁
                except IndexError:
                    row.append(False)
            self.walls.append(row) # 遍历行然后枚举列 加入每一行中

        self.solution = None

    # 预处理边界
    def print(self):
        solution = self.solution[1] if self.solution is not None else None
        print()
        for i, row in enumerate(self.walls):
            for j, col in enumerate(row):
                if col:
                    print("█", end="") # 墙壁继续打印墙壁
                elif (i, j) == self.start:
                    print("A", end="")
                elif (i, j) == self.goal:
                    print("B", end="")
                elif solution is not None and (i, j) in solution:
                    print("*", end="") # 如果点在目标路径中则打印该点
                else:
                    print(" ", end="") # 空的地方继续空着
            print()
        print()


    def neighbors(self, state):
        row, col = state
        candidates = [ # 向上下左右 4 个方向扩展
            ("up", (row - 1, col)),
            ("down", (row + 1, col)),
            ("left", (row, col - 1)),
            ("right", (row, col + 1))
        ]

        result = []
        for action, (r, c) in candidates: # 往上下左右 4 个方向寻找可以进入的点
            # 没有越界 且 扩展的该点不是墙壁
            if 0 <= r < self.height and 0 <= c < self.width and not self.walls[r][c]:
                result.append((action, (r, c)))
        return result


    def solve(self):
        """Finds a solution to maze, if one exists."""

        # Keep track of number of states explored
        self.num_explored = 0

        # 初始化状态
        start = Node(state=self.start, parent=None, action=None)
        frontier = StackFrontier() # 定义栈使用dfs搜索, 这里可换成 QueueFrontier
        frontier.add(start) # 起点入栈, 队列同理

        # 初始化一个空 set
        self.explored = set() # 对走过的点进行标记：为了不重复标记而使用set

        # 找到终点即终止
        while True:

            # 如果搜索过程中栈为空了, 证明不存在一条路径使得 A -> B
            if frontier.empty():
                raise Exception("no solution")

            # 从边界中选择一个节点
            node = frontier.remove()
            self.num_explored += 1 # 已经探索的点的数量 ++, 相当于set 的计数器++

            # 如果这个节点是目标节点'B', 则存在路径
            if node.state == self.goal:
                actions = []
                cells = []
                # while模拟回溯 记录路径
                while node.parent is not None:
                    actions.append(node.action)
                    cells.append(node.state)
                    node = node.parent
                actions.reverse()
                cells.reverse()
                self.solution = (actions, cells)
                return

            # 将该进入栈的节点打上标记
            self.explored.add(node.state)

            # 向4个方向进行扩展
            for action, state in self.neighbors(node.state):
                # 没有在栈中, 且没有被走过
                if not frontier.contains_state(state) and state not in self.explored:
                    child = Node(state = state, parent = node, action = action)
                    frontier.add(child)


    def output_image(self, filename, show_solution=True, show_explored=False):
        from PIL import Image, ImageDraw
        cell_size = 50
        cell_border = 2

        # Create a blank canvas
        img = Image.new(
            "RGBA",
            (self.width * cell_size, self.height * cell_size),
            "black"
        )
        draw = ImageDraw.Draw(img)

        solution = self.solution[1] if self.solution is not None else None
        for i, row in enumerate(self.walls):
            for j, col in enumerate(row):

                # Walls
                if col:
                    fill = (40, 40, 40)

                # Start
                elif (i, j) == self.start:
                    fill = (255, 0, 0)

                # Goal
                elif (i, j) == self.goal:
                    fill = (0, 171, 28)

                # Solution
                elif solution is not None and show_solution and (i, j) in solution:
                    fill = (220, 235, 113)

                # Explored
                elif solution is not None and show_explored and (i, j) in self.explored:
                    fill = (212, 97, 85)

                # Empty cell
                else:
                    fill = (237, 240, 252)

                # Draw cell
                draw.rectangle(
                    ([(j * cell_size + cell_border, i * cell_size + cell_border),
                      ((j + 1) * cell_size - cell_border, (i + 1) * cell_size - cell_border)]),
                    fill=fill
                )

        img.save(filename)

# 命令行参数的个数 != 2 (令人暖心的提示^ ^):提示你要怎么样调用 这段代码
if len(sys.argv) != 2:
    sys.exit("Usage: python maze.py maze.txt")

m = Maze(sys.argv[1]) 
print("Maze:")
m.print() # 把迷宫maze.txt先打印了
print("Solving...") 
m.solve() # 输出路径
print("States Explored:", m.num_explored) # 搜索的状态数
print("Solution:")
m.print()
#m.output_image("maze.png", show_explored=True)