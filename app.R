library(shiny)
knitr::opts_chunk$set(echo = TRUE)
library(heatmaply)
library(gtsummary)
library(dplyr)
# Implement

df = read.csv("concats")
df2 <- df %>% select(c(123, 87:121))
rownames(df2) <- df2$itemLabel
df2 <- df2 %>% select(-itemLabel)

# df_several <- df2[c("Barack Obama", "Hillary Clinton", "Joe Biden", "Elizabeth Warren", "Andrew Cuomo", "Donald Trump", "Lindsey Graham", "George W. Bush"),] 
# # rownames(df_several) <- c("B. Obama", "H. Clinton", "J. Biden", "E. Warren", "A. Cuomo", "D. Trump", "L. Graham", "G. W. Bush")
# colnames(df_several)
# knitr::combine_words(rownames(df_several), before = "'", after = "'")
df2 <- normalize(df2)

data_selection <- function(df2, names, personality){
  my_data <- subset(df2, rownames(df2) %in% names)
  my_data2 <- my_data %>%  select(personality)
  
  mymap <- heatmaply(my_data2,  xlab = "Personality Features",
                     ylab = "Politicians")
  
  
  return(mymap)
}

# data_selection(df2, c("Barack Obama", "Hillary Clinton"), c('neuroticism', 'anxiety'))

# Defining the UI
ui <- fluidPage(
  # Main panel for displaying outputs
  titlePanel("Hello Politicians!"),
  
  
  fluidRow(
    
    

           column(6,
             selectInput("personality", label = "Select personality aspects: ",
                         choices = c('neuroticism', 'anxiety', 'hostility', 'depression', 'self_consciousness', 'immoderation', 'vulnerability', 'extraversion', 'friendliness', 'gregariousness', 'assertiveness', 'activity_level', 'excitement_seeking', 'cheerfulness', 'openness', 'imagination', 'artistic_interests', 'emotionality', 'adventurousness', 'intellect', 'liberalism', 'agreeableness', 'trust', 'morality', 'altruism', 'cooperation', 'modesty', 'sympathy', 'conscientiousness', 'self_efficacy', 'orderliness', 'dutifulness', 'achievement_striving', 'self_discipline', 'cautiousness'), selected = c( 'openness',  'conscientiousness', 'extraversion', 'agreeableness' , 'neuroticism'), multiple = TRUE)
           ),
           column(6,
                  selectInput("names", label = "Select politicians to compare: ",
                              choices = c('Barack Obama', 'Hillary Clinton', 'Bernie Sanders', 'Nancy Pelosi', 'Andrew Cuomo', 'Chuck Schumer', 'Joe Biden', 'Bill de Blasio', 'Elizabeth Warren', 'John Kerry', 'Adam Schiff', 'Pete Buttigieg', 'Chris Murphy', 'Richard Blumenthal', 'Kamala Harris', 'Cory Booker', 'Jerry Brown', 'Gavin Newsom', 'Rahm Emanuel', 'Alexandria Ocasio-Cortez', 'Amy Klobuchar', 'Phil Murphy', 'Ron Wyden', 'Tim Kaine', 'Eric Garcetti', 'Dianne Feinstein', 'Jay Inslee', 'Kirsten Gillibrand', 'John Bel Edwards', 'Bill Clinton', 'Dick Durbin', 'Chris Smith', 'Janet Yellen', 'Sherrod Brown', 'Marty Walsh', 'Terry McAuliffe', 'Michelle Obama', 'Mark Stoops', 'Joe Manchin', 'Chris Coleman', 'Robert Menendez', 'Ralph Northam', 'Xavier Becerra', 'Thomas Perez', 'Claire McCaskill', 'John Hickenlooper', 'Maxine Waters', 'JuliÃ¡n Castro', 'Chris Coons', 'Patrick Leahy', 'Steny Hoyer', 'Elijah Cummings', 'Sylvester Turner', 'Tulsi Gabbard', 'Jimmy Carter', 'Ed Markey', 'Ben Cardin', 'Lori Lightfoot', 'Gretchen Whitmer', 'Mark Dayton', 'David Ige', 'Patty Murray', 'Brian Schatz', 'J.B. Pritzker', 'Ashton Carter', 'Jeh Johnson', 'Tom Wheeler', 'Mitch Landrieu', 'Josh Shapiro', 'Jared Polis', 'Eric Schneiderman', 'Jon Tester', 'Gina Raimondo', 'Eric Swalwell', 'Jim Kenney', 'Maura Healey', 'London Breed', 'Al Franken', 'Robert Johnson', 'Ilhan Omar', 'Keith Ellison', 'Mary Kathryn Heitkamp', 'Muriel Bowser', 'Tim Walz', 'Mike Duggan', 'Richard Cordray', 'Chris Van Hollen', 'Dannel Malloy', 'John Cook', 'Greg Fischer', 'Tom Carper', 'Steve Bullock', 'Anthony Foxx', 'Mike Thompson', 'Jeff Merkley', 'Jerrold Nadler', 'Eric Holder', 'Letitia James', 'Al Gore', 'Stacey Abrams', 'Donald Trump', 'Mike Pompeo', 'Lindsey Graham', 'Mike Pence', 'Mitch McConnell', 'Marco Rubio', 'Ted Cruz', 'John McCain', 'Jeff Sessions', 'Nikki Haley', 'Rand Paul', 'Chris Christie', 'Rex Tillerson', 'Rick Scott', 'Sean Spicer', 'Sarah Sanders', 'John Kasich', 'Greg Abbott', 'Kellyanne Conway', 'John Cornyn', 'Jeb Bush', 'Scott Gottlieb', 'Steven Mnuchin', 'Chuck Grassley', 'Bob Corker', 'Mick Mulvaney', 'Heather Nauert', 'Susan Collins', 'Mark Randall Meadows', 'Rudy Giuliani', 'David Davis', 'Asa Hutchinson', 'Jerome Powell', 'Ron DeSantis', 'Wilbur Ross', 'Bruce Rauner', 'Jeff Flake', 'Tom Cotton', 'Mike Huckabee', 'Roy Cooper', 'Orrin Hatch', 'John Roberts', 'Ivanka Trump', 'Jim Caldwell', 'Matt Bevin', 'Newt Gingrich', 'Lamar Alexander', 'Steve Bannon', 'Jim Jordan', 'Mike DeWine', 'Joe Scarborough', 'Ryan Zinke', 'Kevin Brady', 'Lisa Murkowski', 'Steve King', 'Robert Jones Portman', 'Scott Pruitt', 'Doug Ducey', 'Alex Azar', 'Betsy DeVos', 'Devin Nunes', 'Carly Fiorina', 'Rick Snyder', 'George W. Bush', 'Mitt Romney', 'Kim Reynolds', 'Brett Kavanaugh', 'Jim Bridenstine', 'Ben Sasse', 'Bobby Jindal', 'Reince Priebus', 'Joni Ernst', 'John Boehner', 'Tom Cole', 'James Lankford', 'Trey Gowdy', 'Steve Scalise', 'Bill Haslam', 'Gary Herbert', 'Patrick Joseph Toomey', 'John Delaney', 'Ken Paxton', 'Cory Gardner', 'Chris Sununu', 'Brian Johnson', 'Henry McMaster', 'Anthony Scaramucci', 'Mac Thornberry', 'Paul LePage', 'Roy Blunt', 'Richard M. Burr', 'Kay Ivey', 'Mike Kelly', 'Matt Gaetz', 'Robert Lighthizer', 'Pete Ricketts', 'Tom Price', 'Anthony Kennedy', 'Eric Holcomb', 'Bill Cassidy'), selected = c('Barack Obama','Andrew Cuomo', 'Hillary Clinton','Elizabeth Warren', 'Joe Biden', 'Donald Trump', 'Lindsey Graham', 'George W. Bush'), multiple = TRUE)
           
    ),
    column(12,
           plotlyOutput("plot", width = "100%")
    )
  )
  
  
)

server = function(input, output) {
  output$plot <- renderPlotly(data_selection(df2, input$names, input$personality))
}
shinyApp(ui = ui, server = server)



