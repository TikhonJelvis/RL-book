-- Replace block quotes with quotes using a custom class from Alon.cls
function CodeBlock(block)
   if block.attributes.quoteBy and block.attributes.quoteFrom then
      local by = block.attributes.quoteBy
      local from = block.attributes.quoteFrom

      if FORMAT:match 'latex' then
         return {
            pandoc.RawBlock('latex', '\\begin{VT1}'),
            pandoc.Para(block.text),
            pandoc.RawBlock('latex', '\\VTA{' .. by .. '}{' .. from .. '}'),
            pandoc.RawBlock('latex', '\\end{VT1}'),
         }
      else
         return {
            pandoc.BlockQuote(block.text),
            pandoc.BulletList({
                  {
                     pandoc.Para({
                           pandoc.Str(by .. ", "),
                           pandoc.Emph(from)
                     })
                  }
            })
         }
      end
   end
end
