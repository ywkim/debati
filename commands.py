from discord.ext import commands

class Commands(commands.Cog):
    def __init__(self, bot):
        self.bot = bot

    @commands.command()
    async def ask(self, ctx, *, question):
        # GPT-3를 사용하여 질문에 대한 답변을 생성하고 보내는 코드를 여기에 작성합니다.
        pass

    # 다른 명령어들도 이와 같은 방식으로 추가합니다.

def setup(bot):
    bot.add_cog(Commands(bot))
